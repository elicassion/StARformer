import argparse
import future, pickle, blosc, os
from datetime import datetime
from collections import deque

import math, random
import numpy as np

import cv2
from PIL import Image

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from utils import set_seed, create_dataset, top_k_logits

import gym, dmc2gym

from starformer import Starformer, StarformerConfig



parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--domain', type=str, default='cheetah')
parser.add_argument('--task', type=str, default='run')
parser.add_argument('--frame_stack', type=int, default=3)
parser.add_argument('--action_repeat', type=int, default=4)

parser.add_argument('--seq_len', type=int, default=30)
parser.add_argument('--model_type', type=str, default='star')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--patch_size', type=int, default=7)

parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample from each of the buffers.') # unused, keep for extensibility
parser.add_argument('--data_dir_prefix', type=str, default='./dmc_replay/')

parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-4)

parser.add_argument('--save_model', action='store_true')
parser.add_argument('--save_dir', type=str, default='trained_model')



args = parser.parse_args()

# cartpole -- simple task uses shorter training
if args.domain == "cartpole":
    args.epochs = 5
print ("[<<<", args.model_type, args.domain, args.task, args.data_cl, args.seed, ">>>]")


# make deterministic
set_seed(args.seed)

# DMC env helpers
def rgb2gray(rgb):
    # print(rgb.shape)
    r, g, b = rgb[0], rgb[1], rgb[2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = np.clip(gray, 0, 255)
    # print(gray.shape)
    return gray[np.newaxis,:,:] / 255.

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k // 3,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(rgb2gray(obs))
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(rgb2gray(obs))
        return self._get_obs(), reward, done

    def _get_obs(self):
        assert len(self._frames) == self._k
        return torch.FloatTensor(np.concatenate(list(self._frames), axis=0))




@torch.no_grad()
def sample(model, x, sample=False, top_k=None, actions=None, rewards=None):
    model.eval()
    logits, _, _ = model(x, actions, rewards=rewards)
    
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    # apply softmax to convert to probabilities
    probs = F.softmax(logits, dim=-1)
    # sample from the distribution or take the most likely
    if sample:
        ix = torch.multinomial(probs, num_samples=1)
    else:
        _, ix = torch.topk(probs, k=1, dim=-1)

    x = ix

    return x

class TrainerConfig:
    # optimization parameters, will be overried by given actual parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    
    lr_decay = False
    warmup_tokens = 375e6 
    final_tokens = 260e9 

    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        # define a DMC env
        env = dmc2gym.make(
            domain_name=self.config.args.domain,
            task_name=self.config.args.task,
            seed=self.config.args.seed,
            visualize_reward=False,
            from_pixels=True,
            height=84,
            width=84,
            frame_skip=self.config.args.action_repeat
        )
        # print('Env shape:', env.reset().shape)
        env.seed(self.config.args.seed)
        env.action_space.seed(self.config.args.seed)

        env = FrameStack(env, k=self.config.args.frame_stack)

        def run_epoch(split, epoch_num=0):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            data_len = len(loader)
            pbar = tqdm(enumerate(loader), total=data_len) if is_train else enumerate(loader)
            
            # define continous action padding
            action_pad = torch.zeros(config.batch_size, 1, config.vocab_size, device='cuda', dtype=torch.float32)

            for it, (x, y, r) in pbar:
                x = x.to(self.device) # states
                y = y.to(self.device) # action
                r = r.to(self.device) # reward

                # forward the model
                with torch.set_grad_enabled(is_train):
                    if "rwd" in config.model_type:
                        act_logit, atts, loss = model(x, torch.cat([torch.ones(y.size(0), 1, 1, device=y.device, dtype=torch.long)*config.vocab_size, y[:, :-1]], dim=1), targets=y, 
                                                        rewards=torch.cat([torch.zeros(r.size(0), 1, 1, device=r.device, dtype=torch.long), r[:, :-1]]))
                    else:
                        act_logit, atts, loss = model(x, torch.cat([torch.ones(y.size(0), 1, 1, device=y.device, dtype=torch.long)*config.vocab_size, y[:, :-1]], dim=1), targets=y)
                    
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus

                    if is_train:
                        for p in model.parameters():
                            p.grad = None
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                        optimizer.step()
                        if writer is not None:
                            writer.add_scalar('training_loss', loss.item(), epoch_num * data_len + it)


                if is_train:
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum()
                        if self.tokens < config.warmup_tokens:
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

        self.tokens = 0

        for epoch in range(config.max_epochs):
            run_epoch('train', epoch_num=epoch)
            eval_return, eval_std = self.get_returns(0)
            if writer is not None:
                writer.add_scalar('eval_return', eval_return, epoch)
                writer.add_scalar('eval_std', eval_std, epoch)

            if self.config.args.save_model:
                if not os.path.exists(self.config.args.save_dir):
                    os.makedirs(self.config.args.save_dir)
                raw_model = self.model.module if hasattr(self.model, "module") else self.model
                fn = "_".join([str(x) for x in [self.config.args.model_type, self.config.args.domain, self.config.args.task, 
                    self.config.args.seq_len, self.config.args.seed, self.config.args.patch_size, epoch]])+".pth"
                torch.save(raw_model.state_dict(), os.path.join(self.config.args.save_dir, fn))




    def get_returns(self, ret):
        self.model.train(False)
        # define action padding
        action_pad = np.zeros(self.config.vocab_size, dtype=np.float32)
        T_rewards, T_Qs = [], []
        done = True
        for i in range(10):
            state = env.reset()
            all_states = state.type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
            rtgs = [ret]
            rewards = []
            actions = []

            # padding
            actions += [action_pad.copy()]
            rewards += [0]
            sampled_action = sample(self.model.module, all_states, sample=True,
                                              actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1).unsqueeze(0),
                                              rewards=torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(1).unsqueeze(0) if 'rtg' not in self.config.model_type else torch.tensor(rtgs, dtype=torch.float32).to(self.device).unsqueeze(1).unsqueeze(0))
            j = 0
            while True:
                if done:
                    state, reward_sum, done, prev_attn = env.reset(), 0, False, None
                    all_states = state.type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
                    actions = []
                    rewards = []
                    rtgs = [ret]

                    # padding
                    actions += [self.config.vocab_size]
                    rewards += [0]

                # take a step
                action = sampled_action.cpu().numpy()[0,-1]
                actions += [sampled_action]
                state, reward, done = env.step(action)
                rewards += [reward]
                reward_sum += reward
                state = state.unsqueeze(0).to(self.device)
                rtgs += [rtgs[-1] - reward]
        
                # trunk trajectory
                all_states = torch.cat([all_states, state.unsqueeze(0)], dim=1)
                if all_states.size(1) > self.config.maxT:
                    all_states = all_states[:, -self.config.maxT:]
                    actions = actions[-self.config.maxT:]
                    rewards = rewards[-self.config.maxT:]
                    rtgs = rtgs[-self.config.maxT:]
                j += 1

                if done:
                    T_rewards.append(reward_sum)
                    break
                
                sampled_action = sample(self.model.module, all_states, sample=True, 
                                                   actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1).unsqueeze(0),
                                                   rewards=torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(1).unsqueeze(0) if 'rtg' not in self.config.model_type else torch.tensor(rtgs, dtype=torch.float32).to(self.device).unsqueeze(1).unsqueeze(0))

        # env.close()
        eval_return = sum(T_rewards)/10.
        eval_std = np.std(T_rewards)
        print("eval return: %d, eval std: %f" % (eval_return, eval_std))
        
        self.model.train(True)
        return eval_return, eval_std


class StateActionReturnDatasetDMC(Dataset):
    """
    DMC Dataset
    """
    def __init__(self, args):
        self.args = args
        self.batch_size = args.batch_size
        self.block_size = args.seq_len
        self.dir = args.data_dir_prefix
        self.domain = args.domain
        self.task = args.task

        self.vocab_size = 0
        self.obses = None
        self.actions = None
        self.rewards = None
        self.not_dones = None
        self.done_idxs = None
        self.timesteps = None
        self.load()

    def load(self):
        path = os.path.join(self.dir, '{}_{}_rb.pt'.format(self.domain, self.task))
        payload = torch.load(path, encoding="bytes")

        self.obses = payload[0]
        self.actions = payload[2]
        self.rewards = payload[3]
        self.not_dones = payload[4].reshape(-1)
        self.done_idxs = [0]*len(self.not_dones)
        self.rtgs = [0]*len(self.not_dones)
        self.timesteps = [0]*len(self.not_dones)
        self.vocab_size = len(self.actions[0])

        self.grayscale = Grayscale()
        last_idx = 0
        for idx, not_done in enumerate(self.not_dones):
            if not_done == 0:
                for j in range(last_idx, idx+1):
                    self.done_idxs[j] = idx
                for j in range(idx-1, last_idx-1):
                    self.rtgs[j] = self.rtgs[j+1]+self.rewards[j]
                last_idx = idx + 1
        for j in range(last_idx, len(self.done_idxs)):
            self.done_idxs[j] = len(self.done_idxs)


    def get_reward_stats(self):
        return None, None



    def __getitem__(self, idx):
        block_size = self.block_size
        # done_idx = idx + block_size
        done_idx = min(self.done_idxs[idx], idx+block_size)
        idx = done_idx - block_size

        states = self.grayscale(F.interpolate(torch.tensor(self.obses[idx:done_idx], dtype=torch.float32), size=(84, 84), mode='bilinear').view(block_size, 3, 3, 84, 84)).view(block_size,3,84,84) # (block_size, 4*84*84)
        states = states / 255.
        # states = self.transform(states, img_size[-2:])
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.float32).view(block_size, -1) # (block_size, A)
        if 'dt' in self.args.model_type:
            rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        else:
            rtgs = torch.clamp(torch.tensor(self.rewards[idx:done_idx], dtype=torch.float32).unsqueeze(1), 0, 1)
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)

        return states, actions, rtgs, timesteps

    def __len__(self):
        return len(self.obses) - self.block_size
                        
                              
# run
writer = None

# uncomment for tensorboard
# timestr = datetime.now().strftime("%m-%d-%H-%M-%S")
# writer = SummaryWriter('runs/{}_{}'.format("_".join([str(x) for x in [args.model_type, args.domain, args.task, args.seq_len, args.seed, args.patch_size]]), timestr))

img_size = (4, 84, 84)

train_dataset = StateActionReturnDatasetDMC(args)

# initialize 
mconf = StarformerConfig(train_dataset.vocab_size, img_size = img_size, patch_size = (args.patch_size, args.patch_size), pos_drop=0.1, resid_drop=0.1,
                      N_head=8, D=192, local_N_head=4, local_D=64, model_type=args.model_type, n_layer=6, C=img_size[0], maxT = args.seq_len, action_type="continuous")
model = Starformer(mconf)


tconf = TrainerConfig(max_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, vocab_size=train_dataset.vocab_size, img_size=img_size,
                      lr_decay=True, warmup_tokens=512*20, final_tokens=10*len(train_dataset)*args.seq_len,
                      num_workers=8, seed=args.seed, model_type=args.model_type, maxT = args.seq_len,
                      args=args)
trainer = Trainer(model, train_dataset, None, tconf)

trainer.train()
del train_dataset, model, trainer