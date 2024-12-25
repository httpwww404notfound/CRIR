import argparse
from recommender import RecAgent
from logger import logger
import gc
from envs.myVirTB import myVirTB

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # basic parameters
    parser.add_argument('--device', type=str, default='cuda:0', help='device cuda or cpu or others')
    parser.add_argument('--epsilon', type=float, default=1., help='starting epsilon for ε-greedy exploration')
    parser.add_argument('--discount_factor', type=float, default=0.9, help='hidden dimension of critic networks')
    parser.add_argument('--state_size', type=int, default=50, help='considered users history length')
    parser.add_argument('--algorithm', type=str, default='DDPG_CL', help='chosen algorithm in DDPG_CL DDPG_DIN')
    parser.add_argument('--max_episode_num', type=int, default=20000)
    parser.add_argument('--env', type=str, default='myVirTB')

    # logger setup
    parser.add_argument('--log_base_path', type=str, default='save_data')
    parser.add_argument('--log_file_name', type=str, default='exp1.txt')

    # take effect when environment is discrete
    parser.add_argument('--user_num', type=int, default=500)
    parser.add_argument('--item_num', type=int, default=1000)

    # take effect when applying deep neural network
    parser.add_argument('--embedding_dim', type=int, default=100 ,help='dimension of embedding for users and items')
    parser.add_argument('--actor_hidden_dim', type=int, default=128 ,help='hidden dimension of actor networks')
    parser.add_argument('--actor_learning_rate', type=float, default=0.001 ,help='hidden dimension of actor networks')
    parser.add_argument('--critic_hidden_dim', type=int, default=128, help='hidden dimension of critic networks')
    parser.add_argument('--critic_learning_rate', type=float, default=0.001, help='hidden dimension of critic networks')
    parser.add_argument('--batch_size', type=int, default=32)

    # take effect when applying DDPG
    parser.add_argument('--tau', type=float, default=0.001, help='momentum update rate')
    parser.add_argument('--actor_tau', type=float, default=0.001, help='actor update changing rate')
    parser.add_argument('--actor_lamda', type=float, default=10.0, help='init actor_lamda factor')

    # take effect when applying contrastive learning
    parser.add_argument('--user_buffer_length', type=int, default=20, help='init actor_lamda factor')
    parser.add_argument('--frequency', type=float, default=1.0)
    parser.add_argument('--floor_freq', type=float, default=1.0)
    parser.add_argument('--warm_up_episode', type=int, default='32')

    # take effect when applying cluster
    parser.add_argument('--num_intent_clusters', type=int, default=32)
    parser.add_argument('--seed', type=float, default=0.5)
    parser.add_argument('--cluster_frequency', type=int, default=40)
    parser.add_argument('--max_cluster_pool', type=int, default=2000)

    # model saving and loading setup
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--load_model', type=int, default=0)
    parser.add_argument('--save_model_path', type=str, default='save_weights')

    # replay buffer settings
    parser.add_argument('--replay_memory_size', type=int, default=10000, help='size of replay buffer')
    # when applying learnable buffer
    parser.add_argument('--hidden_buffer_dim', type=int, default=100)

    args = parser.parse_args()

    for i in ['exp1.txt', 'exp2.txt']:
        env = eval(args.env + f'(\'{args.device}\', {args.state_size}, {args.user_buffer_length})')

        logging = logger(args.log_base_path)

        args.log_file_name = i
        agent = RecAgent(env, args, logging)
        agent.train()
        del agent, env, logging
        gc.collect()
