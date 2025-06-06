import numpy as np
from .tree import SumTree, MinTree
import random
import torch

class PriorityExperienceReplay(object):
    """
    apply PER
    """

    def __init__(self, buffer_size, embedding_dim, device):
        self.buffer_size = buffer_size
        self.crt_idx = 0
        self.is_full = False
        self.device = device
        
        '''
            state : (300,), 
            next_state : (300,) 변할 수 잇음, 
            actions : (100,), 
            rewards : (1,), 
            dones : (1,)
        '''
        self.states = np.zeros((buffer_size, 2*embedding_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, embedding_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.next_states = np.zeros((buffer_size, 2*embedding_dim), dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.int_)

        self.sum_tree = SumTree(buffer_size)
        self.min_tree = MinTree(buffer_size)

        self.max_priority = 1.0
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_constant = 0.00001

    def append(self, state, action, reward, next_state, done):
        self.states[self.crt_idx] = state.numpy()
        self.actions[self.crt_idx] = action.numpy()
        self.rewards[self.crt_idx] = reward.numpy()
        self.next_states[self.crt_idx] = next_state.numpy()
        self.dones[self.crt_idx] = done.numpy()

        self.sum_tree.add_data(self.max_priority ** self.alpha)
        self.min_tree.add_data(self.max_priority ** self.alpha)
        
        self.crt_idx = (self.crt_idx + 1) % self.buffer_size
        if self.crt_idx == 0:
            self.is_full = True

    def sample(self, batch_size):
        rd_idx = []
        weight_batch = []
        index_batch = []
        sum_priority = self.sum_tree.sum_all_prioirty()
        
        N = self.buffer_size if self.is_full else self.crt_idx
        min_priority = self.min_tree.min_prioirty() / sum_priority
        max_weight = (N * min_priority) ** (-self.beta)

        segment_size = sum_priority/batch_size
        for j in range(batch_size):
            min_seg = segment_size * j
            max_seg = segment_size * (j + 1)

            random_num = random.uniform(min_seg, max_seg)
            priority, tree_index, buffer_index = self.sum_tree.search(random_num)
            rd_idx.append(buffer_index)

            p_j = priority / sum_priority
            w_j = (p_j * N) ** (-self.beta) / max_weight
            weight_batch.append(w_j)
            index_batch.append(tree_index)
        self.beta = min(1.0, self.beta + self.beta_constant)

        batch_states = torch.from_numpy(self.states[rd_idx]).device(self.device)
        batch_actions = torch.from_numpy(self.actions[rd_idx]).device(self.device)
        batch_rewards = torch.from_numpy(self.rewards[rd_idx]).device(self.device)
        batch_next_states = torch.from_numpy(self.next_states[rd_idx]).device(self.device)
        batch_dones = torch.from_numpy(self.dones[rd_idx]).device(self.device)

        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, torch.tensor(weight_batch, device=self.device), index_batch

    def update_priority(self, priority, index):
        self.sum_tree.update_prioirty(priority ** self.alpha, index)
        self.min_tree.update_prioirty(priority ** self.alpha, index)
        self.update_max_priority(priority ** self.alpha)

    def update_max_priority(self, priority):
        self.max_priority = max(self.max_priority, priority)