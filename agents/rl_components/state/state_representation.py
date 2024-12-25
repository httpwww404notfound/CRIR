import torch
from torch import nn
import numpy as np
from .layers import *

'''
avarage representation
'''
class DRRAveSR(nn.Module):
    def __init__(self, embedding_dim):
        super(DRRAveSR, self).__init__()
        self.embedding_dim = embedding_dim
        self.wav = nn.Conv1d(50, 1, 1)

    def forward(self, x):
        # x[0] user x[1] item
        items_eb = x[1]/self.embedding_dim # item_eb.shape = [1, 50, 100]
        wav = self.wav(items_eb) # wav.shape = [1, 1, 100]
        wav = torch.squeeze(wav, dim=1) # wav.shape = [1, 100]
        x[0] = torch.squeeze(x[0], dim=1)
        user_wav = torch.mul(x[0], wav) # 点乘
        concat = torch.cat([x[0], user_wav, wav],dim=-1) # concat.shape = [1, 300]
        return concat

'''
DRR attentive representation
'''
class DRRAttSR(nn.Module):
    def __init__(self, embedding_dim):
        super(DRRAttSR, self).__init__()
        self.embedding_dim = embedding_dim
        self.wav = nn.Conv1d(1, 1, 1)
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=1)

    def forward(self, x, training=None, mask=None):
        # x[0] user x[1] item
        u_v_att = self.attention([x[0], x[1]])
        u_v_att = torch.mean(u_v_att, dim=1)
        u_v_dot = torch.mul(x[0], u_v_att)
        concat = torch.cat([x[0], u_v_dot, u_v_att])
        concat = torch.flatten(concat)
        return concat

'''
DIN representation
'''
class DeepInterestNetwork(nn.Module):
    def __init__(self, embedding_dim, state_size):
        super().__init__()
        self.attn = AttentionSequencePoolingLayer(max_len=state_size, embedding_dim=embedding_dim)

    def forward(self, x):
        u_v_att, item_weight = self.attn(x[0], x[1])  # [batch_size, 1, 100]

        avg = torch.mean(x[1], dim=-2).unsqueeze(dim=-2)
        avg = avg * x[0]

        out = torch.cat([avg, u_v_att], dim=-1)  # [batch_size, 1, 200]
        return out, item_weight
