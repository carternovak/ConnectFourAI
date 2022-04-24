from cmath import exp
from typing import List
import torch
import random
from torch import nn, rand
from collections import deque
from collections import namedtuple
from ConnectXBoard import *
from ConnectXGym import *
from RandomConnect4Player import *

ExperienceTuple = namedtuple('ExperienceTuple', ['state', 'action', 'next_state', 'reward', 'done'])

class ReplayMemory:
    def __init__(self, memory_size) -> None:
        self.memory = deque(maxlen=memory_size)

    def store(self, state, action, next_state, reward, done):
        self.memory.appendleft(ExperienceTuple(state, action, next_state, reward, done))

    def sample(self, sample_size) -> List[ExperienceTuple]:
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, input_size, action_states) -> None:
        super(DQN, self).__init__()
        self.l1 = nn.Linear(input_size, 20)
        self.l2 = nn.Linear(20, action_states)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.l2(self.relu(self.l1(x)))

# Pseudo Code
# For episode in episodes
class DQNAgent:
    def __init__(self, env, input_size, action_states, batch_size, lr, gamma, epsilon, epsilon_decay = 0.01, min_epsilon=0.1, target_update_rate = 10) -> None:
        self.memory = ReplayMemory(5000)
        self.env = env
        self.input_size = input_size
        self.action_states = action_states
        self.policy_net = DQN(input_size, action_states)
        self.target_net = DQN(input_size, action_states)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss = nn.SmoothL1Loss()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.target_update_rate = target_update_rate

    def predict(self, state, exploration_mode = True):
        exploration_rate = self.epsilon if exploration_mode else 0
        if (random.random() < exploration_rate):
            # Return a random state
            return random.randrange(self.action_states)
        else:
            # Return a state based on the policy
            return torch.argmax(self.policy_net(state))

    def experience_replay(self):
        if (self.batch_size > len(self.memory)):
            return

        experience_batch = self.memory.sample(self.batch_size)

        for experience in experience_batch:
            # Prediction is based on the target net
            # (if done then the future reward is 0 )
            predicted = torch.tensor(experience.reward) + ((self.gamma * self.target_net(experience.next_state).max()) if not experience.done else 0)
            # expected is based on the polic net
            expected = self.policy_net(experience.state)[experience.action]
            #print("predicted")
            #print(predicted)
            #print("expected")
            #print(expected)
            loss = self.loss(predicted, expected)
            loss.backward()
            self.optimizer.step()
        
        self.epsilon = max(self.epsilon * (1 - self.epsilon_decay), self.min_epsilon)
        #transitions = ExperienceTuple(*(zip(*experience_batch)))

        #transition_states = torch.cat(transitions.state)
        #transition_actions = torch.cat(transitions.action)
        #transition_rewards = torch.cat(transitions.reward)



    def train(self, episodes):
        for ep_num in range(episodes):
            done = False
            # Reset the training environment
            # Get the initial state from the environment
            state = self.env.reset()
            # Run through the game until it ends
            while not done:

                action = self.predict(state)
                next_state, reward, done, info = self.env.step(action)
                self.memory.store(state, action, next_state, reward, done)
                state = next_state
                # optimize the model
                self.experience_replay()

            # print(f'Episode {ep_num + 1} Done')
            if ep_num % self.target_update_rate != 0:
                # Update the target net to use the trained policy net
                self.target_net.load_state_dict(self.policy_net.state_dict())


if __name__ == '__main__':
    height = 6
    width = 7
    x = 4
    state_size = height * width
    action_size = width
    board = ConnectXBoard(height=height, width=width, x=x)
    env = ConnectXGym(board, RandomConnect4Player(), 1, -1)
    agent = DQNAgent(env, input_size=state_size, action_states=action_size, batch_size=50, epsilon=.4, gamma=.9, lr=0.01)
    agent.train(50)
