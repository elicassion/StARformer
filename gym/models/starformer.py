from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_


    

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        return F.gelu(input) 

class Attention(nn.Module):
    def __init__(self, config, N_head=6, D=128):
        super().__init__()
        assert D % N_head == 0
        self.config = config
        
#         self.N = N # token_num
        self.N_head = N_head
        
        self.key = nn.Linear(D, D) # D (n_embd) = N_h (n_heads) x D_h (n_headdim)
        self.query = nn.Linear(D, D)
        self.value = nn.Linear(D, D)
        
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resd_drop = nn.Dropout(config.resid_pdrop)
        
        self.proj = nn.Linear(D, D)
    def forward(self, x):
        # x: B * N * D
        B, N, D = x.size()
        
        q = self.query(x.view(B*N, -1)).view(B, N, self.N_head, D//self.N_head).transpose(1, 2)
        k = self.key(x.view(B*N, -1)).view(B, N, self.N_head, D//self.N_head).transpose(1, 2)
        v = self.value(x.view(B*N, -1)).view(B, N, self.N_head, D//self.N_head).transpose(1, 2)
        
        A = F.softmax(q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))), dim=-1)
        A_drop = self.attn_drop(A)
        y = (A_drop @ v).transpose(1, 2).contiguous().view(B, N, D)
        y = self.resd_drop(self.proj(y))
        return y, A

class CausalSelfAttention(nn.Module):
    def __init__(self, config, N_head=6, D=128, T=30):
        super().__init__()
        assert D % N_head == 0
        self.config = config
        
#         self.N = N # token_num
        self.N_head = N_head
        
        self.key = nn.Linear(D, D) # D (n_embd) = N_h (n_heads) x D_h (n_headdim)
        self.query = nn.Linear(D, D)
        self.value = nn.Linear(D, D)
        
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resd_drop = nn.Dropout(config.resid_pdrop)
        
        self.proj = nn.Linear(D, D)
    def forward(self, x, mask=None):
        # x: B * N * D
        B, N, D = x.size()
        
        q = self.query(x.view(B*N, -1)).view(B, N, self.N_head, D//self.N_head).transpose(1, 2)
        k = self.key(x.view(B*N, -1)).view(B, N, self.N_head, D//self.N_head).transpose(1, 2)
        v = self.value(x.view(B*N, -1)).view(B, N, self.N_head, D//self.N_head).transpose(1, 2)

        
        A = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
        if mask is not None:
            A = A.masked_fill(mask[:,:,:N,:N] == 0, float('-inf'))
        A = F.softmax(A, dim=-1)
        A_drop = self.attn_drop(A)
        y = (A_drop @ v).transpose(1, 2).contiguous().view(B, N, D)
        y = self.resd_drop(self.proj(y))
        return y, A
    

    
class PatchEmb(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        D, iD = config.D, config.local_D
        p1, p2 = config.patch_size
        c, h, w = config.img_size

        maxt = config.maxT
        self.patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b t) (h w) (p1 p2 c)', p1 = p1, p2 = p2),
            nn.Linear(p1*p2*c, iD)
        )
        # +1 for mask
        self.action_emb = nn.Embedding(config.vocab_size+1, iD)
        if 'rwd' in config.model_type:
#             self.reward_emb = nn.Embedding(1000, iD)
              self.reward_emb = nn.Sequential(nn.Linear(1, iD), nn.Tanh())
        self.spatial_emb = nn.Parameter(torch.zeros(1, h*w//p1//p2, iD))
        
        
        self.temporal_emb = nn.Parameter(torch.zeros(1, maxt, D))
        if 'xconv' not in config.model_type:
            self.conv_emb = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4, padding=0), nn.ReLU(),
                                     nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                     nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
                                     nn.Flatten(), nn.Linear(3136, D), nn.Tanh())
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'spatial_emb', 'temporal_emb'}
    
    def forward(self, states, actions, rewards=None):
#         print (states.size(), actions.size())
        B, T, C, H, W = states.size()
        local_state_tokens = (self.patch_embedding(states) + self.spatial_emb).reshape(B, T, -1, self.config.local_D)
#         print (self.conv_emb(states.reshape(-1, C, H, W)).reshape(B, T, -1).size())
        if 'xconv' in self.config.model_type:
            global_state_tokens = 0
        else:
            global_state_tokens = self.conv_emb(states.reshape(-1, C, H, W)).reshape(B, T, -1) + self.temporal_emb[:, :T]
        local_action_tokens = self.action_emb(actions.reshape(-1, 1)).reshape(B, T, -1).unsqueeze(2) # B T 1 iD
#         print (local_action_tokens.size(), local_state_tokens.size())
        if 'rwd' in self.config.model_type:
            local_reward_tokens = self.reward_emb(rewards.reshape(-1, 1)).reshape(B, T, -1).unsqueeze(2)
            local_tokens = torch.cat((local_action_tokens, local_state_tokens, local_reward_tokens), dim=2)
        else:
            local_tokens = torch.cat((local_action_tokens, local_state_tokens), dim=2)
        return local_tokens, global_state_tokens, self.temporal_emb[:, :T]
    
class VectorPatchEmb(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        D, iD = config.D, config.local_D
        c = config.state_dim

        maxt = config.maxT
        
        
        # +1 for mask
        self.patch_embedding = nn.Linear(1, iD)
        self.action_emb = nn.Linear(config.vocab_size, iD)
        if 'rwd' in config.model_type:
#             self.reward_emb = nn.Embedding(1000, iD)
              self.reward_emb = nn.Sequential(nn.Linear(1, iD), nn.Tanh())
        self.spatial_emb = nn.Parameter(torch.zeros(1, c, iD))
        self.temporal_emb = nn.Parameter(torch.zeros(1, maxt, D))
        if 'xconv' not in self.config.model_type:
            self.vec_emb = nn.Linear(c, D)
        
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'spatial_emb', 'temporal_emb'}
    
    def forward(self, states, actions, rewards=None):
#         print (states.size(), actions.size())
        B, T, C = states.size()
        local_state_tokens = (self.patch_embedding(states.unsqueeze(-1)).reshape(B*T, C, -1) + self.spatial_emb).reshape(B, T, -1, self.config.local_D)
#         print (self.conv_emb(states.reshape(-1, C, H, W)).reshape(B, T, -1).size())
        if 'xconv' in self.config.model_type:
            global_state_tokens = 0
        else:
            global_state_tokens = self.vec_emb(states.reshape(-1, C)).reshape(B, T, -1) + self.temporal_emb[:, :T]
        local_action_tokens = self.action_emb(actions.reshape(-1, self.config.vocab_size)).reshape(B, T, -1).unsqueeze(2) # B T 1 iD
#         print (local_action_tokens.size(), local_state_tokens.size())
        if 'rwd' in self.config.model_type:
            local_reward_tokens = self.reward_emb(rewards.reshape(-1, 1)).reshape(B, T, -1).unsqueeze(2)
            local_tokens = torch.cat((local_action_tokens, local_state_tokens, local_reward_tokens), dim=2)
        else:
            local_tokens = torch.cat((local_action_tokens, local_state_tokens), dim=2)
        return local_tokens, global_state_tokens, self.temporal_emb[:, :T]
    
    
class _SABlock(nn.Module):
    def __init__(self, config, N_head, D):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.ln2 = nn.LayerNorm(D)
        self.attn = CausalSelfAttention(config, N_head=N_head, D=D)
        self.mlp = nn.Sequential(
                nn.Linear(D, 4*D),
                GELU(),
                nn.Linear(4*D, D),
                nn.Dropout(config.resid_pdrop)
            )

        
    def forward(self, x, mask=None):
        y, att = self.attn(self.ln1(x), mask)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, att
    
    
class SABlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        patch_count = config.state_dim
        self.local_block = _SABlock(config, config.local_N_head, config.local_D)
        self.global_block = _SABlock(config, config.N_head, config.D)
        if 'fusion' in config.model_type:
            self.register_buffer("mask", torch.tril(torch.ones(config.maxT*2, config.maxT*2))
                                     .view(1, 1, config.maxT*2, config.maxT*2))
        else:
            self.register_buffer("mask", torch.tril(torch.ones(config.maxT*2, config.maxT*2))
                                     .view(1, 1, config.maxT*2, config.maxT*2))
            for i in range(0, config.maxT*2, 2):
                self.mask[0, 0, i, i-1] = 1.
        self.local_norm = nn.LayerNorm(config.local_D)
        if 'rwd' in config.model_type:
            total_patch_count = patch_count + 1 + 1
        else:
            total_patch_count = patch_count + 1
        
        self.local_global_proj = nn.Sequential(
                nn.Linear(total_patch_count*config.local_D, config.D), # +1 for action token +1 for reward token
                nn.LayerNorm(config.D)
            )
        
        
    def forward(self, local_tokens, global_tokens, temporal_emb=None):
#         B, T, _ = global_tokens.size()
#         print (local_tokens.size())
        B, T, P, d = local_tokens.size()
        local_tokens, local_att = self.local_block(local_tokens.reshape(-1, P, d))
        local_tokens = local_tokens.reshape(B, T, P, d)
        lt_tmp = self.local_norm(local_tokens.reshape(-1, d)).reshape(B*T, P*d)

        lt_tmp = self.local_global_proj(lt_tmp).reshape(B, T, -1)
        if 'fusion' in self.config.model_type or 'xconv' in self.config.model_type:
            global_tokens += lt_tmp
            if ('xconv' in self.config.model_type) and (temporal_emb is not None):
                global_tokens += temporal_emb
            global_tokens, global_att = self.global_block(global_tokens, self.mask)
            return local_tokens, global_tokens, local_att, global_att
        else:
            if temporal_emb is not None:
                lt_tmp += temporal_emb
            global_tokens = torch.stack((lt_tmp, global_tokens), dim=2).view(B, -1, self.config.D)
            global_tokens, global_att = self.global_block(global_tokens, self.mask)
            return local_tokens, global_tokens[:, 1::2], local_att, global_att
                
                
    
class Starformer(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        # potential setting: D=384 iD=16 patch=(7, 7) then iD x token_num = 16 * 144 = 2304
        D, iD = config.D, config.local_D
#         p1, p2 = config.patch_size
#         c, h, w = config.img_size

        maxt = config.maxT
        
        self.token_emb = VectorPatchEmb(config)
        self.blocks = nn.ModuleList([SABlock(config) for _ in range(config.n_layer)])
        
        self.local_pos_drop = nn.Dropout(config.pos_drop)
        self.global_pos_drop = nn.Dropout(config.pos_drop)
        
        self.ln_head = nn.LayerNorm(config.D)
#         self.head = nn.Linear(config.D, config.vocab_size)
        self.state_head = nn.Linear(config.D, config.state_dim)
        self.action_head = nn.Sequential(
            *([nn.Linear(config.D, config.vocab_size)] + [nn.Tanh()])
        )
        self.reward_head = nn.Linear(config.D, 1)
        
        self.apply(self._init_weights)
        
        print("number of parameters: %d" % sum(p.numel() for p in self.parameters()))
        
    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('token_emb.spatial_emb')
        no_decay.add('token_emb.temporal_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def get_loss(self, pred, target):
        return F.cross_entropy(pred.reshape(-1, pred.size(-1)), target.reshape(-1), reduction='none')

    
    def forward(self, states, actions, rewards=None, target=None, rtgs=None, ts=None, attention_mask=None):
        # actions should be already padded by dataloader

        local_tokens, global_state_tokens, temporal_emb = self.token_emb(states, actions, rewards=rewards)
        local_tokens = self.local_pos_drop(local_tokens)
        if 'xconv' not in self.config.model_type:
            global_state_tokens = self.global_pos_drop(global_state_tokens)
        
        local_atts, global_atts = [], []
        for i, blk in enumerate(self.blocks):
            if i == 0:
                local_tokens, global_state_tokens, local_att, global_att = blk(local_tokens, global_state_tokens, temporal_emb)
            else:
                local_tokens, global_state_tokens, local_att, global_att = blk(local_tokens, global_state_tokens, temporal_emb)
        
        x = self.ln_head(global_state_tokens)
        state_pred = self.state_head(x)
        action_pred = self.action_head(x)
        reward_pred = self.reward_head(x)

        return state_pred, action_pred, reward_pred


    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.config.state_dim)
        actions = actions.reshape(1, -1, self.config.vocab_size)
        rewards = rewards.reshape(1, -1, 1)


        if self.config.maxT is not None:
            states = states[:,-self.config.maxT:]
            actions = actions[:,-self.config.maxT:]
            rewards = rewards[:, -self.config.maxT:]

        state_preds, action_preds, reward_preds = self.forward(
            states, actions, target=None, rewards=rewards)

        return action_preds[0, -1]
    
#------------------------------------------------------------------------
    
        
class StarformerConfig:
    """ based on base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, **kwargs):
        self.vocab_size = vocab_size
        for k,v in kwargs.items():
            setattr(self, k, v)

        assert self.D % self.N_head == 0
        
        
if __name__ == "__main__":
    mconf = StarformerConfig(4, img_size = (4, 84, 84), patch_size = (7, 7), context_length=30, pos_drop=0.1, resid_drop=0.1,
                          N_head=8, D=192, local_N_head=4, local_D=64, model_type='star', max_timestep=100, n_layer=6, C=4, maxT=30)

    model = Starformer(mconf)
    model = model.cuda()
    dummy_states = torch.randn(3, 28, 4, 84, 84).cuda()
    dummy_actions = torch.randint(0, 4, (3, 28, 1), dtype=torch.long).cuda()
    output, atn, loss = model(dummy_states, dummy_actions, None)
    print (output.size(), output)

    dummy_states = torch.randn(3, 1, 4, 84, 84).cuda()
    dummy_actions = torch.randint(0, 4, (3, 1, 1), dtype=torch.long).cuda()
    output, atn, loss = model(dummy_states, dummy_actions, None)
    print (output.size(), output)
    
    for pn, p in model.named_parameters():
        print (pn, p.numel())
#     mconf = StarformerConfig(4, img_size = (4, 84, 84), patch_size = (7, 7), context_length=30, pos_drop=0.1, resid_drop=0.1,
#                           N_head=8, D=192, local_N_head=4, local_D=64, model_type='star', max_timestep=100, n_layer=6, C=4, maxT = 30)

#     model = Starformer(mconf)
#     model = model.cuda()
#     dummy_states = torch.randn(3, 30, 4, 84, 84).cuda()
#     dummy_actions = torch.randint(0, 4, (3, 30, 1), dtype=torch.long).cuda()
#     output, atn, loss = model(dummy_states, dummy_actions, None)
#     print (output.size(), output)
    
#     dummy_states = torch.randn(3, 1, 4, 84, 84).cuda()
#     dummy_actions = torch.randint(0, 4, (3, 1, 1), dtype=torch.long).cuda()
#     output, atn, loss = model(dummy_states, dummy_actions, None)
#     print (output.size(), output)

    
    
#     mconf = StarformerConfig(4, img_size = (4, 84, 84), patch_size = (7, 7), context_length=30, pos_drop=0.1, resid_drop=0.1,
#                           N_head=8, D=192, local_N_head=4, local_D=64, model_type='star_fusion', max_timestep=100, n_layer=6, C=4, maxT=30)
    
#     model = Starformer(mconf)
#     model = model.cuda()
#     dummy_states = torch.randn(3, 28, 4, 84, 84).cuda()
#     dummy_actions = torch.randint(0, 4, (3, 28, 1), dtype=torch.long).cuda()
#     output, atn, loss = model(dummy_states, dummy_actions, None)
#     print (output.size(), output)
    
#     dummy_states = torch.randn(3, 1, 4, 84, 84).cuda()
#     dummy_actions = torch.randint(0, 4, (3, 1, 1), dtype=torch.long).cuda()
#     output, atn, loss = model(dummy_states, dummy_actions, None)
#     print (output.size(), output)