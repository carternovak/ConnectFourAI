from DQNConnect4Player import *
from ConnectXBoard import *
from ConnectXGym import *
from ConnectXDQNTrainer import *


class DuelingDQNs:

    def __init__(self, height = 6, width = 7, x = 4) -> None:
        height = 6
        width = 7
        x = 4
        state_size = height * width
        action_size = width
        board_one = ConnectXBoard(height=height, width=width, x=x)
        env_one = ConnectXGym(board_one,  None, 1, -1)
        self.dqn_one = DQNAgent(env_one, input_size=state_size, action_states=action_size, batch_size=50, epsilon=.4, gamma=.9, lr=0.01)
        board_two = ConnectXBoard(height=height, width=width, x=x)
        env_two = ConnectXGym(board_two,  None, 1, -1)
        self.dqn_two = DQNAgent(env_two, input_size=state_size, action_states=action_size, batch_size=50, epsilon=.4, gamma=.9, lr=0.01)

        self.dqn_one.env.other_player = DQNConnect4Player(self.dqn_two)
        self.dqn_two.env.other_player = DQNConnect4Player(self.dqn_one)
        pass

    def train(self, iterations, episodes_per_iteration):
        # Plan
        # Train on DQN using the other as the "environment"
        # Swap
        # Repeart
        for iteration in range(iterations):
            self.dqn_one.train(episodes_per_iteration)
            self.dqn_two.train(episodes_per_iteration)
            print(f'Iteration {iteration + 1} Complete')
        return

    def get_dqns(self):
        return (self.dqn_one, self.dqn_two)