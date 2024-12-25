import gym
from gym import spaces
import numpy as np
import torch
from envs.virtualTB.model.ActionModel import ActionModel
from envs.virtualTB.model.LeaveModel import LeaveModel
from envs.virtualTB.model.UserModel import UserModel
from envs.virtualTB.utils import *


class VirtualTB(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, device):
        self.n_item = 5
        self.n_user_feature = 88
        self.n_item_feature = 27
        self.max_c = 100
        self.obs_low = np.concatenate(([0] * self.n_user_feature, [0,0,0]))
        self.obs_high = np.concatenate(([1] * self.n_user_feature, [29,9,100]))
        self.observation_space = spaces.Box(low = self.obs_low, high = self.obs_high, dtype = np.int32)
        self.action_space = spaces.Box(low = -1, high = 1, shape = (self.n_item_feature,), dtype = np.float32)
        self.device = device
        self.user_model = UserModel(self.device)
        self.user_action_model = ActionModel()
        self.user_leave_model = LeaveModel()
        if self.device == 'cuda:0':
            self.user_model = self.user_model.cuda()
            self.user_action_model = self.user_action_model.cuda()
            self.user_leave_model = self.user_leave_model.cuda()
        self.user_model.load()
        self.user_action_model.load()
        self.user_leave_model.load()
        self.reset()

    def seed(self, sd = 0):
        torch.manual_seed(sd)

    @property
    def state(self):
        return torch.cat([self.cur_user, self.lst_action, self.total_c.unsqueeze(0)], dim=-1)

    def __user_generator(self):
        # with shape(n_user_feature,)
        user = self.user_model.generate()
        self.__leave = self.user_leave_model.predict(user)
        return user

    def step(self, action):
        # Action: tensor with shape (27, )
        cur_user = self.cur_user.unsqueeze(0)
        total_c = self.total_c.unsqueeze(0).unsqueeze(0)
        self.lst_action = self.user_action_model.predict(cur_user, total_c, action.unsqueeze(0)).detach()[0]
        reward = int(self.lst_action[0])
        self.total_a += reward
        self.total_c = self.total_c + 1
        self.rend_action = deepcopy(self.lst_action)
        done = (self.total_c >= self.__leave)
        if done:
            self.cur_user = self.__user_generator().squeeze().detach()
            self.lst_action = FLOAT([0,0]).to(self.device)
        return self.state, reward, done, {'CTR': self.total_a / self.total_c / 10}

    def reset(self):
        self.total_a = 0
        self.total_c = torch.tensor(0, device=self.device)
        self.cur_user = self.__user_generator().squeeze().detach()
        self.lst_action = FLOAT([0,0]).to(self.device)
        self.rend_action = deepcopy(self.lst_action)
        return self.state

    def render(self, mode='human', close=False):
        print('Current State:')
        print('\t', self.state)
        a, b = np.clip(self.rend_action, a_min = 0, a_max = None)
        print('User\'s action:')
        print('\tclick:%2d, leave:%s, index:%2d' % (int(a), 'True' if self.total_c > self.max_c else 'False', int(self.total_c)))
        print('Total clicks:', self.total_a)
