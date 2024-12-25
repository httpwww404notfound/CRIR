import torch
from torch import nn
import numpy as np
from copy import deepcopy

class CriticNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, act_size, state_ratio):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(act_size+state_ratio*embedding_dim, embedding_dim), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.ReLU())
        # self.fc3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        s = torch.cat(x, dim=-1)
        s = self.fc1(s)
        s = self.fc2(s)
        # s = self.fc3(s)
        return self.out(s)