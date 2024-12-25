import torch
import numpy as np
from .tree_torch import SumTree, MinTree
import random
from copy import deepcopy


class PriorityExperienceReplay(object):
    """
    apply PER
    """

    def __init__(self, buffer_size, state_size, device, user_dim, act_size, type='VTB'):
        self.buffer_size = buffer_size
        self.device = device
        self.crt_idx = 0
        self.is_full = False
        self.data_type = torch.int
        self.state_size = state_size
        self.type = type

        if self.type == 'VTB':
            self.data_type = torch.float32
            self.last_action_index = -1
            self.states = torch.zeros((buffer_size, 1), dtype=torch.int, device=device)
            self.cache_actions = torch.zeros(1, device=device)
            self.cache_next_states = torch.zeros(1, device=device)
        else:
            self.states = torch.zeros((buffer_size, state_size), dtype=torch.int, device=device)

        self.users = torch.zeros((buffer_size, user_dim), dtype=self.data_type, device=device)
        self.actions = torch.zeros((buffer_size, act_size), device=device)
        self.rewards = torch.zeros(buffer_size, device=device)
        self.next_states = torch.zeros((buffer_size, state_size), dtype=self.data_type, device=device)
        self.dones = torch.ones(buffer_size, device=device)

        self.sum_tree = SumTree(buffer_size, device)
        self.min_tree = MinTree(buffer_size, device)

        self.max_priority = 1.0
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_constant = 0.00001

    def append(self, users, state, action, reward, next_state, done):
        self.users[self.crt_idx] = users

        if self.type =='VTB':
            self.states[self.crt_idx] = self.last_action_index
            self.last_action_index = -1 if done else self.crt_idx
        else:
            self.states[self.crt_idx] = state
        self.actions[self.crt_idx] = action
        self.rewards[self.crt_idx] = reward
        self.next_states[self.crt_idx] = next_state
        self.dones[self.crt_idx] = done

        self.sum_tree.add_data(self.max_priority ** self.alpha)
        self.min_tree.add_data(self.max_priority ** self.alpha)

        self.crt_idx = (self.crt_idx + 1) % self.buffer_size
        if self.crt_idx == 0:
            self.is_full = True
            if self.type =='VTB':
                del self.cache_actions
                del self.cache_next_states
                if self.device == 'cuda:0':
                    torch.cuda.empty_cache()
                self.cache_actions = deepcopy(self.actions)
                self.cache_next_states = deepcopy(self.next_states)

    def sample(self, batch_size):
        rd_idx = []
        weight_batch = []
        index_batch = []
        sum_priority = self.sum_tree.sum_all_prioirty()

        N = self.buffer_size if self.is_full else self.crt_idx
        min_priority = self.min_tree.min_prioirty() / sum_priority
        max_weight = (N * min_priority) ** (-self.beta)

        segment_size = sum_priority / batch_size
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

        batch_users = self.users[rd_idx]
        batch_states = self.states[rd_idx]
        batch_actions = self.actions[rd_idx]
        batch_rewards = self.rewards[rd_idx]
        batch_next_states = self.next_states[rd_idx]
        batch_dones = self.dones[rd_idx]

        if self.type == 'VTB':
            return_states = []
            for i in range(batch_states.size(0)):
                temp = self.vtbGetAllAction(batch_states[i])
                return_states.append(temp)
            batch_states = return_states
            return_next_states = deepcopy(batch_states)
            for idx in range(len(return_next_states)):
                temp = torch.cat([batch_actions[idx], batch_next_states[idx]]).unsqueeze(0)
                return_next_states[idx] = torch.cat([return_next_states[idx], temp], dim=0)
            batch_next_states = return_next_states

        return batch_users, batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, weight_batch, index_batch


    def update_priority(self, priority, index):
        self.sum_tree.update_prioirty(priority ** self.alpha, index)
        self.min_tree.update_prioirty(priority ** self.alpha, index)
        self.update_max_priority(priority ** self.alpha)

    def update_max_priority(self, priority):
        self.max_priority = max(self.max_priority, priority)

    def vtbRandomSample(self, batch_size):
        total = self.crt_idx if not self.is_full else self.buffer_size
        indexes = np.random.choice(total, batch_size)
        batch_users = self.users[indexes]
        batch_states = self.states[indexes]

        return_states = []
        for i in range(batch_states.size(0)):
            temp = self.vtbGetAllAction(batch_states[i])
            return_states.append(temp)
        batch_states = return_states

        return batch_users, batch_states

    def vtbGetAllAction(self, state_idx):
        result = []
        idx = state_idx.item()
        if not self.is_full:
            actions = self.actions
            next_states = self.next_states
        else:
            actions = self.cache_actions
            next_states = self.cache_next_states

        while idx != -1:
            temp = torch.cat([actions[idx], next_states[idx]])
            result.append(temp)
            idx = self.states[idx].item()
        result.reverse()
        if len(result) > 0:
            result = torch.cat(result).view(-1, self.actions.size(-1) + self.state_size)
        else:
            result = torch.tensor([], device=self.device)
        return result
