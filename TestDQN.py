from ConnectXDQNTrainer import *
from RandomConnect4Player import *
from DQNConnect4Player import *
from ConnectXBoard import *
from ConnectXGym import *
from ConnectXTester import *
from HumanConnect4Player import *
from DuelingDQNs import *
from MinMaxConnect4Player import *
from AlphaBetaConnect4Player import *
from ConnectXHeuristics import *
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
    agent = DQNAgent(env, board_width=width, board_height=height, action_states=action_size, batch_size=100, epsilon=.999, epsilon_decay=0.01, min_epsilon=0.05, gamma=.5, lr=0.0001)
    training_results = agent.train(10000)
    plt.plot(training_results)
    plt.ylabel('Average Reward')
    plt.xlabel('Epoch')
    plt.show()
    # plt.plot(training_results[200:])
    # plt.ylabel('Average Reward')
    # plt.xlabel('Episode')
    # plt.show()
    print('Agent Trained')
    # agent.policy_net.load_state_dict(torch.load(Path('rand-policy.pt')))
    # agent.target_net.load_state_dict(torch.load(Path('rand-target.pt')))
    torch.save(agent.policy_net.state_dict(), Path('rand-policy.pt'))
    torch.save(agent.target_net.state_dict(), Path('rand-target.pt'))

    # dueling_dqns = DuelingDQNs()
    # dueling_dqns.train(100, 600)
    # dqn_1, dqn_2 = dueling_dqns.get_dqns()
    # dqn_1_player = DQNConnect4Player(dqn_1)
    # dqn_2_player = DQNConnect4Player(dqn_2)

    # torch.save(dqn_1.policy_net.state_dict(), Path('100-600-p1-policy.pt'))
    # torch.save(dqn_1.target_net.state_dict(), Path('100-600-p1-target.pt'))
    # torch.save(dqn_2.policy_net.state_dict(), Path('100-600-p2-policy.pt'))
    # torch.save(dqn_2.target_net.state_dict(), Path('100-600-p2-target.pt'))

    # dqn_1_player = DQNConnect4Player(DQNAgent(
    #     env, board_height=height, board_width=width, action_states=action_size, batch_size=128, epsilon=.999, epsilon_decay=0.01, min_epsilon=0.05, gamma=.8, lr=0.0001,
    #     pre_trained_policy=torch.load(Path('100-600-p1-policy.pt')),pre_trained_target=torch.load(Path('100-600-p1-target.pt'))
    #     ))

    # dqn_2_player = DQNConnect4Player(DQNAgent(
    #     env, board_height=height, board_width=width, action_states=action_size, batch_size=128, epsilon=.999, epsilon_decay=0.01, min_epsilon=0.05, gamma=.8, lr=0.0001,
    #     pre_trained_policy=torch.load(Path('100-600-p2-policy.pt')),pre_trained_target=torch.load(Path('100-600-p2-target.pt'))
    #     ))

    sse_player = AlphaBetaConnect4Player(dist_heuristic, 1, 25)
    lines_player = AlphaBetaConnect4Player(partial_lines_heuristic, -1, 5)
    dqn_player = DQNConnect4Player(agent)
    random_player = RandomConnect4Player()
    human_player = HumanConnect4Player()
    tester = ConnectXTester(dqn_player, 'DQN', lines_player, 'Lines', 10, ConnectXBoard(height=height, width=width, x=x))
    tester.test()