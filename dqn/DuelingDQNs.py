from players.DQNConnect4Player import *
from ConnectXBoard import *
from ConnectXGym import *
from ConnectXDQNTrainer import *
import time


class DuelingDQNs:

    def __init__(self, height = 6, width = 7, x = 4) -> None:
        height = 6
        width = 7
        x = 4
        state_size = height * width
        action_size = width
        board_one = ConnectXBoard(height=height, width=width, x=x)
        env_one = ConnectXGym(board_one,  None, 1, -1)
        self.dqn_one = DQNAgent(env_one, board_height=height, board_width=width, action_states=action_size, batch_size=128, epsilon=.999, epsilon_decay=0.01, min_epsilon=0.05, gamma=.5, lr=0.0001)
        board_two = ConnectXBoard(height=height, width=width, x=x)
        env_two = ConnectXGym(board_two,  None, 1, -1)
        self.dqn_two = DQNAgent(env_two, board_height=height, board_width=width, action_states=action_size, batch_size=128, epsilon=.999, epsilon_decay=0.01, min_epsilon=0.05, gamma=.5, lr=0.0001)

        self.dqn_one.env.other_player = DQNConnect4Player(self.dqn_two)
        self.dqn_two.env.other_player = DQNConnect4Player(self.dqn_one)
        pass

    def train(self, iterations, episodes_per_iteration):
        # Plan
        # Train on DQN using the other as the "environment"
        # Swap
        # Repeart
        start_time = time.time()
        last_time = time.time()
        dqn_one_avg_reward = []
        dqn_two_avg_reward = []
        for iteration in range(iterations):
            dqn_one_results = self.dqn_one.train(episodes_per_iteration)
            dqn_two_results = self.dqn_two.train(episodes_per_iteration)
            dqn_one_avg_reward.append(sum(dqn_one_results)/len(dqn_one_results))
            dqn_two_avg_reward.append(sum(dqn_two_results)/len(dqn_two_results))
            iteration_end_time = time.time()
            print(f'Iteration {iteration + 1} Complete, Time Taken: {iteration_end_time-last_time}s')
            last_time = iteration_end_time
        print(f'{iterations} Iterations Complete, Total Time Taken: {iteration_end_time - start_time}s')
        return dqn_one_avg_reward, dqn_two_avg_reward

    def get_dqns(self):
        return (self.dqn_one, self.dqn_two)