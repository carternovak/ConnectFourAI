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

if __name__ == '__main__':
    height = 6
    width = 7
    x = 4
    state_size = height * width
    action_size = width
    board = ConnectXBoard(height=height, width=width, x=x)

    env = ConnectXGym(board, AlphaBetaConnect4Player(dist_heuristic, -1, 30), 1, -1)
    agent = DQNAgent(env, input_size=state_size, action_states=action_size, batch_size=50, epsilon=.4, gamma=.9, lr=0.01)
    # agent.train(50)
    # print('Agent Trained')

    # dueling_dqns = DuelingDQNs()
    # dueling_dqns.train(50, 50)
    # dqn_1, dqn_2 = dueling_dqns.get_dqns()
    # dqn_1_player = DQNConnect4Player(dqn_1)
    # dqn_2_player = DQNConnect4Player(dqn_2)

    ab_player = AlphaBetaConnect4Player(dist_heuristic, 1, 30)
    lines_player = AlphaBetaConnect4Player(partial_lines_heuristic, -1, 30)
    dqn_player = DQNConnect4Player(agent)
    random_player = RandomConnect4Player()
    human_player = HumanConnect4Player()
    tester = ConnectXTester(lines_player, 'Lines', ab_player, 'ABP', 50, ConnectXBoard(height=height, width=width, x=x))
    tester.test()