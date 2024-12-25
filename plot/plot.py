import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import math


def avg_data(y, stage, total):
    result = []
    num_points = int(total / stage)
    x = [i * stage for i in range(num_points)]
    for e in y:
        result.append([np.mean(e[i * stage : (i + 1) * stage]) for i in range(num_points)])
    return x, result

def get_plot_data(data, factor, stage, total):
    all_data = []
    for lines in data:
        y_value = []
        interaction_count = 0
        for line in lines:
            elements = line.strip().split(sep=', ')
            for element in elements:
                index = element.find(':')
                if index == -1:
                    continue
                name, value = element[:index-1], element[index+2:]
                if name == 'episode_length':
                    interaction_count += eval(value)
                if name == factor:
                    y_value.append(eval(value))
            if len(y_value) >= total:
                break
        print(f'episode_length_count : {interaction_count}')
        all_data.append(y_value)
    all_data = np.array(all_data)

    x_value, y_value = avg_data(all_data, stage, total)
    return x_value, y_value

def add_curve(factor, root_dir, stage):
    file_names =  os.listdir(root_dir)
    file_paths = [os.path.join(root_dir, name) for name in file_names] # file_names

    exp_logs = []
    for path in file_paths:
        lines = open(path, 'r').readlines()
        exp_logs.append(lines)

    total = min(20000, min([len(exp_logs[i]) for i in range(len(exp_logs))]))
    multi_curve = len(exp_logs) > 1

    if factor in ['reward_density', 'con_weight_grad', 'RL_weight_grad']:
        x_value, y_1 = get_plot_data(exp_logs, factor, stage, total)
        x_value, y_2 = get_plot_data(exp_logs, 'episode_length', stage, total)
        y_value = [[y_1[num][i] / y_2[num][i] if y_2[num][i] > 0 else 0 for i in range(len(y_1[num]))] for num in range(len(y_1))]
    else:
        x_value, y_value = get_plot_data(exp_logs, factor, stage, total)
    y_mean = np.mean(y_value, axis=0)

    if multi_curve:
        y_std = np.std(y_value, axis=0)
        y_max, y_min = y_mean + 0.95 * y_std, y_mean - 0.95 * y_std
    else:
        y_max, y_min = None, None

    return x_value, y_mean, y_max, y_min, total, multi_curve


def plot(metric, root_dirs, colors, labels):
    # ['precision', 'cur_point', 'mean_reward', 'q_loss', 'a_loss', 'rewards_statistic', 'SCL Loss', 'ICL Loss', 'UCL Loss', 'epsilon']
    metric_dic = {'episode_reward': (200, 0, 80, 10), 'episode_length': (100, 0, 100, 10),
                     'q_loss': (10, None, None, None), 'a_loss': (10, -50, 20, 10),
                     'reward_density': (100, 0, 100, 4), 'CTR':(100, 0, 1.2, 0.2),
                     'con_weight_grad': (200, None, None, None), 'con_bias_grad': (200, None, None, None),
                     'RL_weight_grad': (200, None, None, None), 'RL_bias_grad': (200, None, None, None)
                }
    max_total = 0
    stage, bot, top, step = metric_dic[metric]
    for idx, root_dir in enumerate(root_dirs):
        x_value, y_mean, y_max, y_min, total, multi_curve =add_curve(metric, root_dir, stage)

        max_total = max(total, max_total)
        plt.plot(x_value, y_mean, color=colors[idx], label=labels[idx])
        if multi_curve:
            plt.fill_between(x_value, y_max, y_min, alpha=0.5, facecolor=colors[idx])
    my_x_ticks = np.arange(0, max_total + 1, max_total // 5)
    plt.xticks(my_x_ticks)
    if metric in ['q_loss', 'con_weight_grad', 'con_bias_grad', 'RL_weight_grad', 'RL_bias_grad']:
        plt.yscale("log")
    else:
        my_y_ticks = np.arange(bot, top, step)
        plt.yticks(my_y_ticks)
    plt.xlabel('episode')
    plt.ylabel(metric)# 'log mean square grad'
    plt.legend()
    plt.savefig('./figs/episode_reward.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    pool = {'red':'#e75840', 'blue':'#628cee', 'green': '#7FFF00', 'purple':'#8A2BE2', 'orange':'#FFA500', 'pink':'#FFC0CB', 'cyan': '#00FFFF'}

    root_dirs = ['../save_data/myVirTB/DDPG_CL', '../save_data/myVirTB/DDPG_DIN']
    colors = [pool['red'], pool['blue'], pool['green'], pool['purple'], pool['orange'], pool['pink'], pool['cyan']]

    labels = ['CRIR', 'CRIR w/o CL']

    plot('episode_reward', root_dirs, colors, labels) # or CTR

