import numpy as np
import os
import torch
import math

from agents.abcAgent import abcModel
from agents.rl_components.per_replay.replay_buffer_torch import PriorityExperienceReplay
from agents.rl_components.actor import ActorNetwork
from agents.rl_components.critic import CriticNetwork
from agents.rl_components.state.state_representation import DeepInterestNetwork
from embedding import *

class DDPG_DIN(abcModel):
    def __init__(self, env, args):
        super().__init__()
        self.env = env
        self.is_test = False
        self.device = args.device
        self.save = args.save
        self.save_model_path = args.save_model_path
        self.batch_size = args.batch_size

        self.embedding_dim = args.embedding_dim
        self.actor_hidden_dim = args.actor_hidden_dim
        self.actor_learning_rate = args.actor_learning_rate
        self.critic_hidden_dim = args.critic_hidden_dim
        self.critic_learning_rate = args.critic_learning_rate
        self.discount_factor = args.discount_factor
        self.tau = 0.001
        self.actor_tau = 0.001
        self.actor_lamda = args.actor_lamda

        # ε-greedy exploration hyperparameter
        self.epsilon = args.epsilon
        self.epsilon_decay = (self.epsilon - 0.1) / 50000
        self.std = 1.5

        # actor critic
        self.actor = ActorNetwork(self.embedding_dim, self.actor_hidden_dim, self.env.act_size, 2)
        self.actor_target = ActorNetwork(self.embedding_dim, self.actor_hidden_dim, self.env.act_size, 2)
        self.critic = CriticNetwork(self.embedding_dim, self.actor_hidden_dim, self.env.act_size, 2)
        self.critic_target = CriticNetwork(self.embedding_dim, self.actor_hidden_dim, self.env.act_size, 2)
        self.critic_loss = torch.nn.MSELoss(reduction='none')

        if args.env == 'myVirTB':
            self.state_encoder = statePreEncoder(self.env.state_history_dim, self.embedding_dim)
            self.user_encoder = vtbUserEncoder(self.env.n_user_feature, self.embedding_dim)
            self.sr = DeepInterestNetwork(self.embedding_dim, args.state_size)

        if args.device[:4] == 'cuda':
            self.actor, self.actor_target, self.critic, self.critic_target\
                = self.actor.cuda(), self.actor_target.cuda(), self.critic.cuda(), self.critic_target.cuda()
            self.state_encoder, self.user_encoder, self.sr\
                = self.state_encoder.cuda(), self.user_encoder.cuda(), self.sr.cuda()

        # PER
        self.buffer = PriorityExperienceReplay(args.replay_memory_size, self.env.state_review_dim,
                                    user_dim=self.env.n_user_feature, act_size=self.env.act_size, device=self.device)
        self.epsilon_for_priority = 1e-6

        # optimizers
        self.critic_optimizer = torch.optim.Adam([{'params': self.critic.parameters()},
                                                  {'params': self.sr.parameters()},
                                                  {'params': self.user_encoder.parameters()},
                                                  {'params': self.state_encoder.parameters()}],
                                                 lr=self.actor_learning_rate)
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=self.actor_learning_rate)

    def select_action(self, states, user):
        states = self.pad_state(states, self.env.state_size)
        with torch.no_grad():
            user_eb = self.user_encoder(user.unsqueeze(dim=0))
            states_pre_eb = self.state_encoder(states.unsqueeze(dim=0))
            state_rep, state_weight = self.sr([torch.unsqueeze(user_eb, dim=0), states_pre_eb])

            action = self.actor(state_rep)

            # ε-greedy exploration
            if self.epsilon > np.random.uniform() and not self.is_test:
                self.epsilon = self.epsilon - (self.epsilon_decay if self.epsilon > 0.1 else 0)
                rand = torch.normal(0, self.std, size=action.shape, device=self.device)
                action = action + rand

            action = torch.squeeze(action)

        return action

    def update(self, **kwargs):
        add_statistic = kwargs['add_statistic'] # {'a_loss':0, 'q_loss':0}

        if self.buffer.crt_idx > self.batch_size or self.buffer.is_full:
            batch_user, batch_history, batch_actions, batch_rewards, batch_next_history, batch_dones, weight_batch, index_batch = \
                self.buffer.sample(self.batch_size)
            weight_batch = torch.tensor(weight_batch, device=self.device)

            # actions sequence padding
            batch_history = torch.cat([self.pad_state(i, 50) for i in batch_history]).view(self.batch_size,self.env.state_size, -1)
            batch_next_history = torch.cat([self.pad_state(i, 50) for i in batch_next_history]).view(self.batch_size,self.env.state_size,-1)

            batch_user_emb = self.user_encoder(batch_user).unsqueeze(dim=1)
            batch_history_emb = self.state_encoder(batch_history)
            batch_state_rep, _ = self.sr([batch_user_emb, batch_history_emb])

            with torch.no_grad():
                # next state info
                next_states_eb = self.state_encoder(batch_next_history)
                batch_next_states_rep, _ = self.sr([batch_user_emb, next_states_eb])
                # Set TD targets
                target_next_action = self.actor_target(batch_next_states_rep)
                qs = self.critic([target_next_action, batch_next_states_rep])
                target_qs = self.critic_target([target_next_action, batch_next_states_rep])
                # Double Q method
                min_qs = torch.min(torch.cat([target_qs, qs], dim=1), dim=1)[0]
                td_targets = self.calculate_td_target(batch_rewards, min_qs, batch_dones)
                td_targets = torch.tensor(td_targets, device=self.device)

            # Update critic network SRM embeddings
            outputs = torch.squeeze(self.critic([batch_actions, batch_state_rep.squeeze()]))
            critic_loss = self.critic_loss(outputs, td_targets)
            critic_weighted_loss = torch.mean(critic_loss * weight_batch)
            add_statistic['q_loss'] = critic_weighted_loss.item()

            self.critic_optimizer.zero_grad()
            critic_weighted_loss.backward()
            RL_state_weight_grad, RL_state_bias_grad = self.log_grad()
            add_statistic['RL_weight_grad'], add_statistic['RL_bias_grad'] = add_statistic['RL_weight_grad'] + RL_state_weight_grad, add_statistic['RL_bias_grad'] + RL_state_bias_grad
            self.critic_optimizer.step()

            # Update priority
            td_errors = torch.abs(td_targets - outputs).detach()
            for (e, i) in zip(td_errors, index_batch):
                self.buffer.update_priority(e.item() + self.epsilon_for_priority, i)

            # Update actor network
            with torch.no_grad():
                batch_user_emb = self.user_encoder(batch_user).unsqueeze(dim=1)
                batch_history_emb = self.state_encoder(batch_history)
                batch_state_rep, _ = self.sr([batch_user_emb, batch_history_emb])

            actor_loss = -self.critic([self.actor(batch_state_rep), batch_state_rep]).mean()
            add_statistic['a_loss'] = actor_loss.item()
            self.actor_lamda = self.actor_tau * critic_weighted_loss.item() + (1 - self.actor_tau) * self.actor_lamda
            actor_loss = math.exp(-self.actor_lamda ** 2) * actor_loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # momentum update target network
            self.momentum_update()

        return add_statistic

    def buffer_add(self, user, action, reward, next_states, done):
        next_states_added = next_states[-1, -self.env.state_review_dim:]
        self.buffer.append(user, None, action, reward, next_states_added, done)

    def momentum_update(self):
        # soft target network update
        c_omega = self.actor.parameters()
        t_omega = self.actor_target.parameters()
        for c_para, t_para in zip(c_omega, t_omega):
            t_para.data = self.tau * c_para.data + (1 - self.tau) * t_para.data

        c_omega = self.critic.parameters()
        t_omega = self.critic_target.parameters()
        for c_para, t_para in zip(c_omega, t_omega):
            t_para.data = self.tau * c_para.data + (1 - self.tau) * t_para.data

    def calculate_td_target(self, rewards, q_values, dones):
        y_t = []
        for i in range(q_values.shape[0]):
            y_t.append(rewards[i] + (1 - dones[i]) * (self.discount_factor * q_values[i]))
        return y_t

    def pad_state(self, tensor: torch.tensor, max_len, dim=-2, fill_num=0.0):
        l = dim + 1 if dim >= 0 else -dim
        if len(tensor.size()) < l:
            pad_size = torch.Size([max_len, self.env.state_history_dim])
            tensor = torch.full(pad_size, fill_num, device=self.device)
            return tensor
        else:
            pad_len = max_len - tensor.size(dim)

        if pad_len > 0:
            pad_tensor_size = torch.as_tensor(tensor.size())
            pad_tensor_size[dim] = pad_len
            pad_size = pad_tensor_size.tolist()
            pad_tensor = torch.full(pad_size, fill_num, device=self.device)
            tensor = torch.cat([tensor, pad_tensor], dim=dim)
        elif pad_len < 0:
            pad_len = -pad_len
            remain = [i for i in range(pad_len, tensor.size(dim))]
            remain = torch.tensor(remain, device=self.device, dtype=torch.int)
            tensor = torch.index_select(tensor, dim=dim, index=remain)

        return tensor

    def save(self, episode):
        self.save_model(self.save_model_path, f'actor_{episode + 1}.pth', f'critic_{episode + 1}.pth',
                        f'user_{episode + 1}.pth', f'state_{episode + 1}.pth', f'sr_{episode + 1}.pth')

    def load(self, episode):
        self.load_model(self.save_model_path, f'actor_{episode + 1}.pth', f'critic_{episode + 1}.pth',
                        f'user_{episode + 1}.pth', f'state_{episode + 1}.pth', f'sr_{episode + 1}.pth')

    def save_model(self, base_dir, actor, critic, uEmb, iEmb, sr):
        self.actor.save_weights(os.path.join(base_dir, actor))
        self.critic.save_weights(os.path.join(base_dir, critic))
        torch.save(self.user_encoder.state_dict(), os.path.join(base_dir, uEmb))
        torch.save(self.state_encoder.state_dict(), os.path.join(base_dir, iEmb))
        torch.save(self.sr.state_dict(), os.path.join(base_dir, sr))

    def load_model(self, base_dir, actor, critic, uEmb, iEmb, sr):
        self.actor.load_weights(os.path.join(base_dir, actor))
        self.critic.load_weights(os.path.join(base_dir, critic))
        self.user_encoder.load_state_dict(torch.load(os.path.join(base_dir, uEmb)))
        self.state_encoder.load_state_dict(torch.load(os.path.join(base_dir, iEmb)))
        self.sr.load_state_dict(torch.load(os.path.join(base_dir, sr)))

    def log_grad(self):
        with torch.no_grad():
            mean_square_weight = torch.mean((self.state_encoder.item_encoder.weight.grad)**2)
            mean_square_bias = torch.mean((self.state_encoder.item_encoder.bias.grad)**2)

        return mean_square_weight, mean_square_bias
