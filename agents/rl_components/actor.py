from torch import nn
import torch


class ActorNetwork(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, act_size, state_ratio):
        super(ActorNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_ratio * embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_size),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class ActorProbNetwork(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, act_size, state_ratio):
        super(ActorProbNetwork, self).__init__()
        self.fc1 = nn.Linear(state_ratio * embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, act_size)
        self.fc4 = nn.Linear(hidden_dim, act_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(x)
        mean = torch.tanh(self.fc3(x))
        log_std = torch.tanh(self.fc4(x))

        return mean, log_std