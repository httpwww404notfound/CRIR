from .virtualTB.envs.virtualTB import *
from envs.abcEnv import abcEnv
import torch

# # 定义一个装饰器函数，它接受一个函数作为参数
# def prevent_override(func):
#     # 定义一个新的函数，它接受任意数量的位置参数和关键字参数
#     def wrapper(*args, **kwargs):
#         # 获取调用该函数的对象，即self参数
#         self = args[0]
#         # 判断该对象是否是子类的实例
#         if isinstance(self, Child):
#             # 如果是子类的实例，那么获取父类的reset方法
#             parent_reset = super(Child, self).reset
#             # 调用父类的reset方法，传入相同的参数
#             parent_reset(*args, **kwargs)
#         else:
#             # 如果不是子类的实例，那么直接调用原函数，传入相同的参数
#             func(*args, **kwargs)
#     # 返回新的函数
#     return wrapper

class myVirTB(VirtualTB, abcEnv):
    '''
    state : tensor of concatenated action, lst_action and total_c history
    对外的state是action, lst_action and total_c合并的张量的前
    '''

    def __init__(self, device, state_size, user_buf_size=20):
        super(myVirTB, self).__init__(device)
        self.state_size = state_size
        self.state_history = []
        self.state_review_dim = 2 + 1
        self.state_history_dim = self.n_item_feature + self.state_review_dim
        self.user_buf_size = user_buf_size
        self.user_buffer = []
        self.act_size = self.n_item_feature

    def step_(self, action):
        action = action.squeeze()
        state, reward, done, _ = super(myVirTB, self).step(action)
        total_c = self.total_c.unsqueeze(0)
        add_history = torch.cat([action, self.lst_action, total_c])
        self.state_history.append(add_history)
        self.state_history = self.state_history[-self.state_size:]
        state_history = torch.cat(self.state_history, dim=0).view(-1, self.state_history_dim)
        return state_history, reward, done

    # @prevent_override
    def reset_(self):
        super(myVirTB, self).reset()
        cur_user = self.cur_user.clone().detach()
        self.state_history = []
        if len(self.user_buffer) < self.user_buf_size:
            self.user_buffer.append(cur_user)
        else:
            self.user_buffer = self.user_buffer[1:]
            self.user_buffer.append(cur_user)
        return torch.tensor([], dtype=torch.float, device=self.device), cur_user

    # def sample_history_users(self, sample_num):
    #     idx = np.random.sample(len(self.user_buffer), max(len(self.user_buffer),sample_num), replace=False)
    #     users = torch.cat(self.user_buffer).view(sample_num, -1)
    #     users = users[idx]
    #     return users

    def generate_users(self, gen_num):
        users = []
        for i in range(gen_num):
            user = self.user_model.generate()
            users.append(user.squeeze())
        # users = torch.cat(users, dim=0).view(gen_num, -1)
        return users

if __name__ == '__main__':

    env = myVirTB('cuda:0', 50)
    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.low)
    print(env.observation_space.high)
    state = env.reset_()
    while True:
        env.render()
        action = env.action_space.sample()
        state, reward, done, info = env.step_(action)

        if done: break
    env.render()