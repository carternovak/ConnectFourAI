from ConnectXDQNTrainer import *
from ConnectXGym import *
from DuelingDQNs import *
from SelfPlay import self_play, true_self_play
from GreedyConnect4Player import GreedyConnect4Player

from RandomConnect4Player import *
from DQNConnect4Player import *
from HumanConnect4Player import *
from MinMaxConnect4Player import *
from AlphaBetaConnect4Player import *

from ConnectXTester import *

from ConnectXBoard import *
from ConnectXHeuristics import *

from THUNDERDOME import ThunderDome

from pathlib import Path
import matplotlib.pyplot as plt

if __name__ == '__main__':
    height = 6
    width = 7
    x = 4
    state_size = height * width
    action_size = width
    board = ConnectXBoard(height=height, width=width, x=x)

    env = ConnectXGym(board, RandomConnect4Player(), 1, -1)
    agent = DQNAgent(env, conv_model=True, board_width=width, board_height=height, action_states=action_size, batch_size=100, epsilon=.999, epsilon_decay=0.01, min_epsilon=0.05, gamma=.5, lr=0.0001)
    # training_results = agent.train(500)
    # plt.plot(training_results)
    # plt.ylabel('Average Reward')
    # plt.xlabel('Epoch')
    # plt.show()
    # # plt.plot(training_results[200:])
    # # plt.ylabel('Average Reward')
    # # plt.xlabel('Episode')
    # # plt.show()
    # print('Agent Trained')
    # agent.policy_net.load_state_dict(torch.load(Path('rand-policy.pt')))
    # agent.target_net.load_state_dict(torch.load(Path('rand-target.pt')))
    # torch.save(agent.policy_net.state_dict(), Path('rand-policy.pt'))
    # torch.save(agent.target_net.state_dict(), Path('rand-target.pt'))

    # dueling_dqns = DuelingDQNs(conv=True, height=height, width=width)
    # dqn_one_avg_reward, dqn_two_avg_reward = dueling_dqns.train(15, 500)
    # # plt.plot(dqn_one_avg_reward)
    # # plt.plot(dqn_two_avg_reward)
    # # plt.ylabel('Average Reward')
    # # plt.xlabel('Epoch')
    # # plt.show()
    # dqn_1, dqn_2 = dueling_dqns.get_dqns()
    # dqn_1_player = DQNConnect4Player(dqn_1)
    # dqn_2_player = DQNConnect4Player(dqn_2)

    # torch.save(dqn_1.policy_net.state_dict(), Path('15-500-conv-p1-policy.pt'))
    # torch.save(dqn_1.target_net.state_dict(), Path('15-500-conv-p1-target.pt'))
    # torch.save(dqn_2.policy_net.state_dict(), Path('15-500-conv-p2-policy.pt'))
    # torch.save(dqn_2.target_net.state_dict(), Path('15-500-conv-p2-target.pt'))

    # dqn_1_player = DQNConnect4Player(DQNAgent(
    #     env, conv_model=True, board_height=height, board_width=width, action_states=action_size, batch_size=128, epsilon=.999, epsilon_decay=0.01, min_epsilon=0.05, gamma=.8, lr=0.0001,
    #     pre_trained_policy=torch.load(Path('5-500-conv-p1-policy.pt')), pre_trained_target=torch.load(Path('5-500-conv-p1-target.pt'))
    #     ))

    # dqn_2_player = DQNConnect4Player(DQNAgent(
    #     env, board_height=height, board_width=width, action_states=action_size, batch_size=128, epsilon=.999, epsilon_decay=0.01, min_epsilon=0.05, gamma=.8, lr=0.0001,
    #     pre_trained_policy=torch.load(Path('100-600-p2-policy.pt')),pre_trained_target=torch.load(Path('100-600-p2-target.pt'))
    #     ))

    # dqn_one_agent = DQNAgent(env, conv_model=True, board_width=width, board_height=height, action_states=action_size, batch_size=100, epsilon=.999, epsilon_decay=0.01, min_epsilon=0.05, gamma=.5, lr=0.0001)
    # dqn_two_agent = DQNAgent(env, conv_model=False, board_width=width, board_height=height, action_states=action_size, batch_size=100, epsilon=.999, epsilon_decay=0.01, min_epsilon=0.05, gamma=.5, lr=0.0001)
    # test_thunderdome = ThunderDome({'DQN_CNN':dqn_one_agent, 'DQN_NOT_CNN':dqn_two_agent}, {'RAND': RandomConnect4Player(),'GREEDY': GreedyConnect4Player()})

    # test_thunderdome.thunderdome(15, 500)

    # thunderdome_players = test_thunderdome.get_trained_models()

    # dqn_cnn_player = DQNConnect4Player(thunderdome_players['DQN_CNN'])
    # dqn_norm_player = DQNConnect4Player(thunderdome_players['DQN_NOT_CNN'])

    # board1 = ConnectXBoard()
    # env1 = ConnectXGym(board1, RandomConnect4Player(), 1, -1)
    # rand_player_one = DQNAgent(env1, width, height, action_size, conv_model = True, batch_size=128, lr=0.005, gamma=0.7, epsilon=0.5, epsilon_decay=0.001, min_epsilon=0.05)
    # rand_player_one.train(200)

    # board2 = ConnectXBoard()
    # env2 = ConnectXGym(board2, RandomConnect4Player(), 1, -1)
    # rand_player_two = DQNAgent(env2, width, height, action_size, conv_model = True, batch_size=128, lr=0.005, gamma=0.7, epsilon=0.5, epsilon_decay=0.001, min_epsilon=0.05)
    # rand_player_two.train(200)

    # def compare(model1, model2):
    #     for p1, p2 in zip(model1.parameters(), model2.parameters()):
    #         if p1.data.ne(p2.data).sum() > 0:
    #             return False
    #     return True

    # self_play_agent = self_play(ConnectXBoard(), 5, episodes_per_generation=1000, learning_episodes=1000)
    # self_player = DQNConnect4Player(self_play_agent)
    true_self_play_agent = true_self_play(ConnectXBoard(), 5, episodes_per_generation=1000, random_episodes=1000)
    true_self_player = DQNConnect4Player(true_self_play_agent)
    sse_player = AlphaBetaConnect4Player(dist_heuristic, 1, 75)
    lines_player = AlphaBetaConnect4Player(partial_lines_heuristic, -1, 75)
    dqn_player = DQNConnect4Player(agent)
    random_player = RandomConnect4Player()
    human_player = HumanConnect4Player()
    # tester = ConnectXTester(DQNConnect4Player(rand_player_one), 'Rand 1', HumanConnect4Player(), 'Human', 5, ConnectXBoard(height=height, width=width, x=x), pause = False, render = 'none')
    # tester.test()
    # tester = ConnectXTester(DQNConnect4Player(rand_player_two), 'Rand 2', HumanConnect4Player(), 'Human', 5, ConnectXBoard(height=height, width=width, x=x), pause = False, render = 'none')
    # tester.test()

    # print(compare(rand_player_one.policy_net, rand_player_two.policy_net))

    tester = ConnectXTester(sse_player, 'SSE', true_self_player, 'True Self Play', 50, ConnectXBoard(height=height, width=width, x=x), pause = False, render = 'none')
    tester.test()