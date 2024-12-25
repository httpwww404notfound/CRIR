import os.path
import sys
from tqdm import tqdm
from agents import *

class RecAgent:

    def __init__(self, env, args, logger):
        self.env = env
        self.args = args
        self.logger = logger
        self.model = getattr(sys.modules[__name__], args.algorithm)(env, args)
        self.max_episode_num = args.max_episode_num
        self.save = args.save
        self.log_file_name = self.args.log_file_name

    def train(self):
        self.logger.create_path(os.path.join(self.args.env, self.args.algorithm))
        if self.args.load_model > 0:
            self.model.load_model(self.args.load_model)
            print('Completely load weights!')

        description = self.args.algorithm
        for episode in tqdm(range(self.max_episode_num), desc=description, leave=True, ncols=100, unit='episode'):
            add_statistic, steps, episode_reward, click = self.train_a_step(episode)

            self.logger.log_line(self.log_file_name,
                f'{episode}/{self.max_episode_num}, episode_reward : {episode_reward}, ' +
                f'episode_length : {steps}, CTR : {click / steps}'
            )

            add_logs = self.process_added_statistic(add_statistic)
            self.logger.log_line(self.log_file_name, add_logs, step_line=True)

        if self.save:
            self.model.save(self.max_episode_num)

    def train_a_step(self, episode):
        # statistics
        episode_reward, click = 0, 0
        add_statistic = {'a_loss': 0, 'q_loss': 0, 'SCL Loss': 0, 'ICL Loss': 0, 'UCL Loss': 0,
                         'SCL Counter': 0, 'ICL Counter': 0, 'UCL Counter': 0,
                         'RL_weight_grad': 0, 'RL_bias_grad': 0,
                         'con_weight_grad': 0, 'con_bias_grad': 0
                         }
                         #

        states, user = self.env.reset_()
        steps, done = 0, False
        while not done:
            action_prob = self.model.select_action(states, user)
            if self.args.algorithm in ['PPO', 'PPO_CL']:
                action = action_prob[0]
            else:
                action = action_prob
            next_states, reward, done = self.env.step_(action)
            if self.args.algorithm == 'NICF':
                self.model.buffer_add(states, action_prob, reward, next_states, done)
            else:
                self.model.buffer_add(user, action_prob, reward, next_states, done)
            add_statistic = self.model.update(episode=episode, steps=steps, done=done, add_statistic=add_statistic)

            steps = steps + 1
            episode_reward += reward
            click += int(reward > 0)
            states = next_states

        return add_statistic, steps, episode_reward, click

    def process_added_statistic(self, add_statistic):
        add_logs = ''

        if 'a_loss' in add_statistic.keys():
            a_loss = add_statistic['a_loss']
            add_logs += f', a_loss : {a_loss}'
        if 'q_loss' in add_statistic.keys():
            q_loss = add_statistic['q_loss']
            add_logs += f', q_loss : {q_loss}'
        if 'SCL Loss' in add_statistic.keys():
            add_statistic['SCL Loss'] = 0 if add_statistic['SCL Counter'] == 0 else add_statistic['SCL Loss'] / add_statistic['SCL Counter']
            cl_1_counter = add_statistic['SCL Counter']
            cl_1_mean = add_statistic['SCL Loss']
            add_logs += f', SCL Loss : {cl_1_counter}×{cl_1_mean}'
        if 'ICL Loss' in add_statistic.keys():
            add_statistic['ICL Loss'] = 0 if add_statistic['ICL Counter'] == 0 else add_statistic['ICL Loss'] / add_statistic['ICL Counter']
            cl_2_counter = add_statistic['ICL Counter']
            cl_2_mean = add_statistic['ICL Loss']
            add_logs += f', ICL Loss : {cl_2_counter}×{cl_2_mean}'
        if 'SCL Loss' in add_statistic.keys():
            add_statistic['UCL Loss'] = 0 if add_statistic['UCL Counter'] == 0 else add_statistic['UCL Loss'] / add_statistic['UCL Counter']
            cl_3_counter = add_statistic['UCL Counter']
            cl_3_mean = add_statistic['UCL Loss']
            add_logs += f', UCL Loss : {cl_3_counter}×{cl_3_mean}'

        if 'con_weight_grad' in add_statistic.keys():
            con_weight_grad = add_statistic['con_weight_grad']
            add_logs += f', con_weight_grad : {con_weight_grad}'
        if 'con_bias_grad' in add_statistic.keys():
            con_bias_grad = add_statistic['con_bias_grad']
            add_logs += f', con_bias_grad : {con_bias_grad}'
        if 'RL_weight_grad' in add_statistic.keys():
            RL_weight_grad = add_statistic['RL_weight_grad']
            add_logs += f', RL_weight_grad : {RL_weight_grad}'
        if 'RL_bias_grad' in add_statistic.keys():
            RL_bias_grad = add_statistic['RL_bias_grad']
            add_logs += f', RL_bias_grad : {RL_bias_grad}'

        return add_logs


    def test(self):
        pass
