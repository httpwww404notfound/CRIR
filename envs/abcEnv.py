from abc import ABCMeta, abstractmethod


class abcEnv(object):
    __meta_class__ = ABCMeta

    def __init__(self):
        self.user_buffer = None
        self.n_user_feature = None
        self.state_size = None
        self.state_review_dim = None
        self.state_history_dim = None
        self.act_size = None
        self.user_buf_size = None
        self.user_buffer = None

    @ abstractmethod
    def generate_users(self, max_num):
        pass

    @abstractmethod
    def step_(self, action):
        pass

    @abstractmethod
    def reset_(self):
        pass


