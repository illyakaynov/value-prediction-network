import argparse
import go_vncdriver
import tensorflow as tf
from envs import create_env
import subprocess as sp
import util
import model
import numpy as np
from worker import new_env

parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('-gpu', '--gpu', default=0, type=int, help='Number of GPUs')
parser.add_argument('-r', '--remotes', default=None,
                    help='The address of pre-existing VNC servers and rewarders to use'
                         '(e.g. -r vnc://localhost:5900+15900,vnc://localhost:5901+15901).')
parser.add_argument('-e', '--env-id', type=str, default="maze",
                    help="Environment id")
parser.add_argument('-a', '--alg', type=str, default="VPN", help="Algorithm: [A3C | Q | VPN]")
parser.add_argument('-mo', '--model', type=str, default="CNN", help="Name of model: [CNN | LSTM]")
parser.add_argument('-ck', '--checkpoint', type=str, default="", help="Path of the checkpoint")
parser.add_argument('-n', '--n-play', type=int, default=1000, help="Num of play")
parser.add_argument('--eps', type=float, default=0.0, help="Epsilon-greedy")
parser.add_argument('--config', type=str, default="", help="config xml file for environment")
parser.add_argument('--seed', type=int, default=0, help="Random seed")

# Hyperparameters
parser.add_argument('-g', '--gamma', type=float, default=0.98, help="Discount factor")
parser.add_argument('--dim', type=int, default=64, help="Number of final hidden units")
parser.add_argument('--f-num', type=str, default='32,32,64', help="num of conv filters")
parser.add_argument('--f-stride', type=str, default='1,1,2', help="stride of conv filters")
parser.add_argument('--f-size', type=str, default='3,3,4', help="size of conv filters")
parser.add_argument('--f-pad', type=str, default='SAME', help="padding of conv filters")

# VPN parameters
parser.add_argument('--branch', type=str, default="4,4,4", help="branching factor")


def evaluate(env, agent, num_play=3000, eps=0.0):
    # a = env.visualize()
    # a.show()
    stats = []
    params = {'num_goals': env.config.object['num_goal'],
              'branch': str(agent.branch),
              'time': env.config.maze['time'],
              'branch_train': agent.train_branch}
    env.max_history = num_play
    for iter in range(0, num_play):
        last_state = env.reset()
        last_features = agent.get_initial_features()
        last_meta = env.meta()
        score = 0
        step = 0
        while True:
            # import pdb; pdb.set_trace()
            if eps == 0.0 or np.random.rand() > eps:
                fetched = agent.act(last_state, last_features,
                                    meta=last_meta)
                if agent.type == 'policy':
                    action, features = fetched[0], fetched[2:]
                else:
                    action, features = fetched[0], fetched[1:]
            else:
                act_idx = np.random.randint(0, env.action_space.n)
                action = np.zeros(env.action_space.n)
                action[act_idx] = 1
                features = []

            state, reward, terminal, info, _ = env.step(action.argmax())
            last_state = state
            last_features = features
            last_meta = env.meta()
            if terminal:
                break

            score += reward
            step += 1
        stats.append({**params, 'score': score,
                      'episode_length': step})
        if env.reward_history[0] < 0:
            pass
    return stats


def run(envs=None):
    stats = []

    args = parser.parse_args()
    args.task = 0
    args.f_num = util.parse_to_num(args.f_num)
    args.f_stride = util.parse_to_num(args.f_stride)
    args.f_size = util.parse_to_num(args.f_size)
    args.branch = util.parse_to_num(args.branch)

    env = new_env(args)
    if envs is None or not envs:
        envs = [env]

    args.meta_dim = 0 if env.meta() is None else len(env.meta())
    device = '/gpu:0' if args.gpu > 0 else '/cpu:0'
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    branches = [
        [4, 4, 4],
        [4, 4, 4, 4],
        [4, 1, 4, 1, 4],
        # [4, 4, 4, 4, 4],
        # [1],
        # [1, 1, 1],
        # [1, 1, 1, 1, 1, 1],
        # [1, 1, 1, 1, 1, 1, 1, 1, 1],
        # [4, 4, 4, 1],
        # [4, 4, 4, 1, 1],
        # [4, 4, 4, 1, 1, 1],
        # [1, 4, 4, 4],
        # [1, 1, 1, 4, 4, 4],
        # [1, 1, 1, 4, 4, 4],
    ]
    paths = [f'/home/ikaynov/Repositories/value-prediction-network/Experiments/{x}/best'
             for x in
             [
                 's10_t20_g8_444',
                 's10_t20_g8_4444',
                 # 's10_t20_g8_44444',
             ]]
    for ck in paths:
        for branch_type in branches:
            config = tf.ConfigProto(device_filters=device,
                                    gpu_options=gpu_options,
                                    allow_soft_placement=True)
            tf.reset_default_graph()
            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())
                if args.alg == 'A3C':
                    model_type = 'policy'
                elif args.alg == 'Q':
                    model_type = 'q'
                elif args.alg == 'VPN':
                    model_type = 'vpn'
                else:
                    raise ValueError('Invalid algorithm: ' + args.alg)
                with tf.device(device):

                    # np.random.seed(args.seed)

                    with tf.variable_scope("local/learner"):
                        agent = eval("model." + args.model)(env.observation_space.shape,
                                                            env.action_space.n, type=model_type,
                                                            gamma=args.gamma,
                                                            dim=args.dim,
                                                            f_num=args.f_num,
                                                            f_stride=args.f_stride,
                                                            f_size=args.f_size,
                                                            f_pad=args.f_pad,
                                                            branch=branch_type,
                                                            meta_dim=args.meta_dim)
                        agent.train_branch = str([int(x) for x in list(ck.split('/')[-2].split('_')[-1])])
                        print("Num parameters: %d" % agent.num_param)
                    saver = tf.train.Saver()
                    saver.restore(sess, ck)

                    for i, env in enumerate(envs):
                        run_stats = evaluate(env, agent, args.n_play, eps=args.eps)
                        stats += run_stats
    return stats


import copy

if __name__ == "__main__":
    from maze import MazeSMDP
    from bs4 import BeautifulSoup

    config = open('config/collect_deterministic.xml').read()
    config = BeautifulSoup(config, "lxml")
    values = list(range(1, 11))

    mazes = []
    # for v in values:
    #     copy_config = copy.copy(config)
    #     copy_config.object['num_goal'] = v
    #     mazes.append(MazeSMDP(config=copy_config))
    stats = run(mazes)

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = pd.DataFrame(stats)
    # df.groupby(by=['num_goals', 'branch']).mean()
    # ax = sns.violinplot(x='num_goals', y='score', data=df)
    # sns.lineplot(x="num_goals", y="score",
    #              err_style="bars", ci=68, data=df)
    #
    # plt.show()
    # fig = plt.Figure(figsize=(10, 10))
    # ax = fig.add_subplot()
    # g = sns.catplot(x="branch", y="score", palette="YlGnBu_d",
    #                  # height=6, aspect=.75,
    #                  kind="boxen", data=df, ax=ax)
    # g.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    # fig.show()
    import plotly.express as px

    df.sort_values(by='branch', inplace=True, ascending=False)
    fig = px.box(df, x="branch", y="score", points="outliers", notched=False, color='branch_train', )
    # fig.show()

    # format the layout
    # fig.update_layout(yaxis=dict(range=[-1, 13]),),
    fig.show()
