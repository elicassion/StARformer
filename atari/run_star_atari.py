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

import atari_py

from starformer import Starformer, StarformerConfig



parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--seq_len', type=int, default=30)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--model_type', type=str, default='star')
parser.add_argument('--num_steps', type=int, default=500000)
parser.add_argument('--num_buffers', type=int, default=50)
parser.add_argument('--game', type=str, default='Breakout')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--patch_size', type=int, default=7)
# 
parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample from each of the buffers.')
parser.add_argument('--data_dir_prefix', type=str, default='./dqn_replay/')

parser.add_argument('--save_model', action='store_true')
parser.add_argument('--save_dir', type=str, default='trained_model')
args = parser.parse_args()


# make deterministic
set_seed(args.seed)


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
                fn = "_".join([str(x) for x in [self.config.args.model_type, self.config.args.game, 
                    self.config.args.seq_len, self.config.args.seed, self.config.args.patch_size, epoch]])+".pth"
                torch.save(raw_model.state_dict(), os.path.join(self.config.args.save_dir, fn))




    def get_returns(self, ret):
        self.model.train(False)
        envargs = EnvArgs(self.config.game.lower(), self.config.seed, self.config.img_size[-2:])
        env = Env(envargs)
        env.eval()

        T_rewards, T_Qs = [], []
        done = True
        for i in range(10):
            state = env.reset()
            all_states = state.type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
            rtgs = [ret]
            rewards = []
            actions = []

            # padding
            actions += [self.config.vocab_size]
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

        env.close()
        eval_return = sum(T_rewards)/10.
        eval_std = np.std(T_rewards)
        print("eval return: %d, eval std: %f" % (eval_return, eval_std))
        
        self.model.train(True)
        return eval_return, eval_std


    
class Env():
    def __init__(self, args):
        self.device = args.device
        self.ale = atari_py.ALEInterface()
        self.ale.setInt('random_seed', args.seed)
        self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
        self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
        self.ale.setInt('frame_skip', 0)
        self.ale.setBool('color_averaging', False)
        self.ale.loadROM(atari_py.get_game_path(args.game))  # ROM loading must be done after setting options
        actions = self.ale.getMinimalActionSet()
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.lives = 0  # Life counter (used in DeepMind training)
        self.life_termination = False  # Used to check if resetting only from loss of life
        self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)
        self.training = True  # Consistent with model training mode
        self.img_size = args.img_size

    def _get_state(self):
        state = cv2.resize(self.ale.getScreenGrayscale(), self.img_size, interpolation=cv2.INTER_LINEAR)
        return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(*self.img_size, device=self.device))

    def reset(self):
        if self.life_termination:
            self.life_termination = False  # Reset flag
            self.ale.act(0)  # Use a no-op after loss of life
        else:
            # Reset internals
            self._reset_buffer()
            self.ale.reset_game()
            # Perform up to 30 random no-ops before starting
            for _ in range(random.randrange(30)):
                self.ale.act(0)  # Assumes raw action 0 is always no-op
                if self.ale.game_over():
                    self.ale.reset_game()
        # Process and return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        self.lives = self.ale.lives()
        return torch.stack(list(self.state_buffer), 0)

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = torch.zeros(2, *self.img_size, device=self.device)
        reward, done = 0, False
        for t in range(4):
            reward += self.ale.act(self.actions.get(action))
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()
            if done:
                break
        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)
        # Detect loss of life as terminal in training mode
        if self.training:
            lives = self.ale.lives()
            if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
                self.life_termination = not done  # Only set flag when not truly done
                done = True
            self.lives = lives
        # Return state, reward, done
        return torch.stack(list(self.state_buffer), 0), reward, done

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return len(self.actions)

    def render(self):
        cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()

class EnvArgs:
    def __init__(self, game, seed, img_size):
        self.device = torch.device('cuda')
        self.seed = seed
        self.max_episode_length = 108e3
        self.game = game
        self.history_length = 4
        self.img_size = img_size

        



class StateActionReturnDataset(Dataset):

    def __init__(self, data, seq_len, actions, done_idxs, rwds, timesteps, img_size=(4, 84, 84)):        
        self.seq_len = seq_len
        self.vocab_size = max(actions) + 1
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rwds = rwds
        self.timesteps = timesteps
        self.transform = F.interpolate
    
    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        seq_len = self.seq_len
        done_idx = idx + seq_len
        for i in self.done_idxs:
            if i > idx: # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - seq_len
        states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).view(seq_len, 4, 84, 84) # (seq_len, 4*84*84)
        states = states / 255.
        states = self.transform(states, img_size[-2:])
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1) # (seq_len, 1)
        rwds = torch.clamp(torch.tensor(self.rwds[idx:done_idx], dtype=torch.float32).unsqueeze(1), 0, 1)
 

        return states, actions, rwds
                        
                              
# run
# returns is step-wise return or rtg, depending on the model type
obss, actions, returns, done_idxs, _ = create_dataset(args.num_buffers, args.num_steps, args.game, args.data_dir_prefix, args.trajectories_per_buffer, args)
writer = None

# uncomment for tensorboard
# timestr = datetime.now().strftime("%m-%d-%H-%M-%S")
# writer = SummaryWriter('runs/{}_{}'.format("_".join([str(x) for x in [args.model_type, args.game, args.seq_len, args.seed, args.patch_size]]), timestr))

img_size = (4, 84, 84)

train_dataset = StateActionReturnDataset(obss, args.seq_len, actions, done_idxs, returns, img_size)

# initialize 
mconf = StarformerConfig(train_dataset.vocab_size, img_size = img_size, patch_size = (args.patch_size, args.patch_size), pos_drop=0.1, resid_drop=0.1,
                      N_head=8, D=192, local_N_head=4, local_D=64, model_type=args.model_type, n_layer=6, C=img_size[0], maxT = args.seq_len)
model = Starformer(mconf)


tconf = TrainerConfig(max_epochs=args.epochs, batch_size=args.batch_size, learning_rate=6e-4, vocab_size=train_dataset.vocab_size, img_size=img_size,
                      lr_decay=True, warmup_tokens=512*20, final_tokens=10*len(train_dataset)*args.seq_len,
                      num_workers=8, seed=args.seed, model_type=args.model_type, game=args.game, maxT = args.seq_len,
                      args=args)
trainer = Trainer(model, train_dataset, None, tconf)

trainer.train()
del train_dataset, model, trainer