import numpy as np
import os
import torch
import math

from agents.abcAgent import abcModel
from agents.rl_components.contrastive.losses import NCELoss, ICL_NCELoss
from agents.rl_components.contrastive.seq_operation import seq_operate
from agents.rl_components.per_replay.replay_buffer_torch import PriorityExperienceReplay
from agents.rl_components.actor import ActorNetwork
from agents.rl_components.critic import CriticNetwork
from agents.rl_components.state.state_representation import DeepInterestNetwork
from embedding import vtbUserEncoder, statePreEncoder

class DDPG_CL(abcModel):
    def __init__(self, env, args):
        super().__init__()
        self.env = env
        self.args = args
        self.is_test = False
        self.device = args.device
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
        self.epsilon_decay = (self.epsilon - 0.1) / 60000
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
        self.icl_optim = torch.optim.Adam(self.state_encoder.parameters(), lr=self.actor_learning_rate)

        # contrastive parameters
        self.frequency = args.frequency
        self.floor_freq = args.floor_freq
        self.fre_decay = (self.frequency - self.floor_freq) / (12000 * 10)
        self.warm_up_episode= args.warm_up_episode
        self.get_pair = seq_operate(args.state_size)
        self.cf_criterion = NCELoss(0.1, self.device)
        self.icl_criterion = ICL_NCELoss(self.device)

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
        episode, steps, add_statistic = kwargs['episode'], kwargs['steps'], kwargs['add_statistic']

        if self.buffer.crt_idx > self.batch_size or self.buffer.is_full:
            batch_user, batch_history, batch_actions, batch_rewards, batch_next_history, batch_dones, weight_batch, index_batch = \
                self.buffer.sample(self.batch_size)
            weight_batch = torch.tensor(weight_batch, device=self.device)

            # contrastive_learning before actor and critic
            cl_loss_1, con_state_weight_grad, con_state_bias_grad = self.contrastive_learning(episode, episode*200+steps, batch_user, batch_history)
            add_statistic['SCL Counter'], add_statistic['SCL Loss'] = add_statistic['SCL Counter'] + int(cl_loss_1 > 0), add_statistic['SCL Loss'] + cl_loss_1
            add_statistic['con_weight_grad'], add_statistic['con_bias_grad'] = add_statistic['con_weight_grad']+con_state_weight_grad, add_statistic['con_bias_grad']+con_state_bias_grad

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
            add_statistic['RL_weight_grad'], add_statistic['RL_bias_grad'] = add_statistic['RL_weight_grad']+RL_state_weight_grad, add_statistic['RL_bias_grad']+RL_state_bias_grad
            self.critic_optimizer.step()

            # Update priority
            # td_errors = torch.abs(td_targets).detach()
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

    def save_model(self, root_dir, actor, critic, uEmb, iEmb, sr):
        save_dir = os.path.join(root_dir, self.args.env, self.args.algorithm)
        self.actor.save_weights(os.path.join(save_dir, actor))
        self.critic.save_weights(os.path.join(save_dir, critic))
        torch.save(self.user_encoder.state_dict(), os.path.join(save_dir, uEmb))
        torch.save(self.state_encoder.state_dict(), os.path.join(save_dir, iEmb))
        torch.save(self.sr.state_dict(), os.path.join(save_dir, sr))

    def load_model(self, root_dir, actor, critic, uEmb, iEmb, sr):
        save_dir = os.path.join(root_dir, self.args.env, self.args.algorithm)
        self.actor.load_weights(os.path.join(save_dir, actor))
        self.critic.load_weights(os.path.join(save_dir, critic))
        self.user_encoder.load_state_dict(torch.load(os.path.join(save_dir, uEmb)))
        self.state_encoder.load_state_dict(torch.load(os.path.join(save_dir, iEmb)))
        self.sr.load_state_dict(torch.load(os.path.join(save_dir, sr)))


    def contrastive_learning(self, episode, steps, included_users, included_states):
        """
        insert contrastive learning procedure into RL process
        """
        state_weight_grad, state_bias_grad = 0, 0
        if self.frequency > self.floor_freq:
            self.frequency -= self.fre_decay
        if self.frequency < np.random.rand():
            return 0, 0, 0
        # sample several users and its related items
        con_users, con_states = included_users, included_states# self.buffer.vtbRandomSample(self.batch_size * 2)

        con_users = torch.cat([con_users, included_users], dim=0)
        con_states = con_states + included_states

        # generate lambda factor
        cl_loss = 0
        # begin contrastive
        if episode > self.warm_up_episode:
            cl_loss, state_weight_grad, state_bias_grad = self._ICL_multi_pair_contrastive_learning(con_users, con_states, 1/3)
            cl_loss = cl_loss.item()
        return cl_loss, state_weight_grad, state_bias_grad

    def _ICL_multi_pair_contrastive_learning(self, batch_user, batch_states, lamda):
        """
        contrastive learning given a batch of pair sequences
        batch_user: batch_size * num_user_features
        batch_states: list of tensor(seq_len * num_state_features) with length batch_size
        lamda: loss weight factor
        """
        batch_states = [i.tolist() for i in batch_states]
        valid_ids = [idx for idx, i in enumerate(batch_states) if len(i) > 1]
        batch_user = batch_user[valid_ids]
        batch_states = [batch_states[i] for i in valid_ids]
        seq_len = [min(len(i), self.env.state_size) for i in batch_states]

        padding_batch_states = self.get_pair.padding(batch_states, pad_value=[0.0 for _ in range(self.env.state_history_dim)]) # [batch_size  max_len]
        user_emb = self.user_encoder(batch_user) # [batch_size  dim]
        states_embs = self.state_encoder(torch.tensor(padding_batch_states, device=self.device))

        # item_weight [batch_size max_len 1]
        _, states_weight= self.sr([torch.unsqueeze(user_emb, dim=1), states_embs])

        # rank [batch_size max_len]
        rank = torch.argsort(states_weight, dim=1, descending=True).tolist()
        # to avoid padded 'id zero items' rank ahead real item ids
        depad_states_ids = [r[-seq_len[i]:] for i, r in enumerate(padding_batch_states)]

        ranked_ids, k_list = self.get_pair.items_contrast_ids(depad_states_ids, rank)
        max_pad_len = 2 + self.env.state_size - self.env.state_size // 2

        padding_ranked_ids = self.get_pair.reverse_padding(ranked_ids, max_pad_len, pad_value=[0.0 for _ in range(self.env.state_history_dim)])
        padding_ranked_emb = self.state_encoder(torch.tensor(padding_ranked_ids, device=self.device)) # [batch_size pad_len dim]

        # weight information generated
        # weight_list = [[1/math.sqrt(i+1)] + [1/pad_len for _ in range(pad_len-1)] for i in k_list]
        weight_list = [1/math.sqrt(i+1) for i in k_list]
        # m = 0
        # for i in range(2,26):
        #     m = m + 1 / math.sqrt(i)
        # m = m / 25
        # weight_list = [m for i in range(len(k_list))]
        ranked_len = [min(max_pad_len, len(i)) for i in ranked_ids]
        cl_loss = lamda * self.icl_criterion(padding_ranked_emb, ranked_len, weight_list)

        self.icl_optim.zero_grad()
        cl_loss.backward()
        state_weight_grad, state_bias_grad =self.log_grad()
        self.icl_optim.step()
        return cl_loss, state_weight_grad, state_bias_grad


    def log_grad(self):
        with torch.no_grad():
            mean_square_weight = torch.mean((self.state_encoder.item_encoder.weight.grad)**2)
            mean_square_bias = torch.mean((self.state_encoder.item_encoder.bias.grad)**2)

        return mean_square_weight, mean_square_bias
