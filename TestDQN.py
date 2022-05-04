from ConnectXDQNTrainer import *
from ConnectXGym import *
from DuelingDQNs import *
from SelfPlay import self_play, true_self_play

from RandomConnect4Player import *
from DQNConnect4Player import *
from HumanConnect4Player import *
from MinMaxConnect4Player import *
from AlphaBetaConnect4Player import *

from ConnectXTester import *

from ConnectXBoard import *
from ConnectXHeuristics import *

from ModelLoading import save_model

if __name__ == '__main__':
    height = 6
    width = 7
    x = 4
    state_size = height * width
    action_size = width
    board = ConnectXBoard(height=height, width=width, x=x)

    env = ConnectXGym(board, AlphaBetaConnect4Player(respective_powered, -1, 125), 1, -1)
    agent = DQNAgent(env, conv_model = True, board_width = width, board_height = height, action_states = action_size, batch_size = 32, epsilon = .999, epsilon_decay = 0.001, min_epsilon = 0.01, gamma = .9, lr = 0.001)
    training_results = agent.train(2500)

    save_model(agent, '2500-eps-respective_powered-125-depth')

    sse_player = AlphaBetaConnect4Player(dist_heuristic, 1, 75)
    lines_player = AlphaBetaConnect4Player(partial_lines_heuristic, -1, 25)
    dqn_player = DQNConnect4Player(agent)
    random_player = RandomConnect4Player()
    human_player = HumanConnect4Player()

    tester = ConnectXTester(random_player, 'Rand 1', random_player, 'Rand 2', 500, ConnectXBoard(height=height, width=width, x=x), pause = False, render = 'none')
    tester.test()