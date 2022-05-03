from typing import List
from sympy import Q
import torch
import random
from torch import nn, rand
from collections import deque
from collections import namedtuple
from ConnectXBoard import *
from ConnectXGym import *
from RandomConnect4Player import *
import time

ExperienceTuple = namedtuple('ExperienceTuple', ['state', 'action', 'next_state', 'reward', 'done'])

class ReplayMemory:
    def __init__(self, memory_size) -> None:
        self.memory = deque(maxlen = memory_size)

    def store(self, state, action, next_state, reward, done):
        self.memory.appendleft(ExperienceTuple(state, action, next_state, reward, done))

    def sample(self, sample_size) -> List[ExperienceTuple]:
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, input_size, action_states, dropout = None) -> None:
        super(DQN, self).__init__()
        self.l1 = nn.Linear(input_size, 50)
        self.l2 = nn.Linear(50, 100)
        self.l3 = nn.Linear(100, 150)
        self.l4 = nn.Linear(150, 100)
        self.l5 = nn.Linear(100, 50)
        self.l6 = nn.Linear(50, action_states)
        self.relu = nn.ReLU()
        if dropout:
            self.dropout = nn.Dropout(p = dropout)
        else:
            self.dropout = nn.Dropout(p = 0)

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.dropout(x)
        x = self.relu(self.l2(x))
        x = self.dropout(x)
        x = self.relu(self.l3(x))
        x = self.dropout(x)
        x = self.relu(self.l4(x))
        x = self.dropout(x)
        x = self.relu(self.l5(x))
        x = self.dropout(x)
        x = self.l6(x)
        return x

class ConvDQN(nn.Module):

    def __init__(self, height, width, action_states, kernel_size=2, stride=1) -> None:
        super(ConvDQN, self).__init__()
        self.height = height
        self.width = width
        self.conv1 = nn.Conv2d(1, 16, kernel_size, stride)
        self.bn16 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size, stride)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size, stride)
        self.bn32 = nn.BatchNorm2d(32)

        # SOURCE: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        def conv2d_size_out(size, kernel_size = kernel_size, stride = stride):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        
        convw = conv2d_size_out(conv2d_size_out(width))#conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(width))))
        convh = conv2d_size_out(conv2d_size_out(height))#conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(height))))

        linear_input_size = convw * convh * 32
        # END
        self.l1 = nn.Linear(linear_input_size, 80)
        # self.l2 = nn.Linear(80, 80)
        # self.l3 = nn.Linear(80, 40)
        # self.l4 = nn.Linear(40, 20)
        self.final = nn.Linear(80, action_states)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn16(self.conv1(x)))
        x = self.relu(self.bn32(self.conv2(x)))
        # x = self.relu(self.bn32(self.conv3(x)))
        # x = self.relu(self.bn32(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.l1(x))
        # x = self.relu(self.l2(x))
        # x = self.relu(self.l3(x))
        # x = self.relu(self.l4(x))
        # Don't use ReLu fo the last one
        x = self.final(x)
        return x

# Pseudo Code
# For episode in episodes
class DQNAgent:
    def __init__(self, env, board_width, board_height, action_states, batch_size, lr, gamma, epsilon, epsilon_decay = 0.01, min_epsilon=0.1, target_update_rate = 500, pre_trained_target = None, pre_trained_policy = None, conv_model = False) -> None:
        self.memory = ReplayMemory(50000)

        self.env = env
        self.board_width = board_width
        self.board_height = board_height

        input_size = board_width * board_height

        self.input_size = input_size
        self.action_states = action_states
        self.conv_model = conv_model

        if (self.conv_model):
            self.policy_net = ConvDQN(board_width, board_height, action_states)
            self.target_net = ConvDQN(board_width, board_height, action_states)
        else:
            self.policy_net = DQN(input_size, action_states)
            self.target_net = DQN(input_size, action_states)

        if (pre_trained_policy and pre_trained_target):
            self.policy_net.load_state_dict(pre_trained_policy)
            self.target_net.load_state_dict(pre_trained_target)
        else :
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        self.target_net.eval()
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
        self.loss = nn.MSELoss()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.target_update_rate = target_update_rate

    def get_move_private(self, state):
        if (self.conv_model):
            state = state.view(1, self.board_height, self.board_width)
        return self.predict(state, False)[0][0]

    def predict(self, state, exploration_mode = True):
        exploration_rate = self.epsilon if exploration_mode else 0
        
        if (self.conv_model):
            top_row = torch.abs(state[0][self.board_height - 1])
        else:
            top_row = torch.abs(state[-self.board_width:])

        illegal_move_penalty = -9999999
        illegal_move_mask = top_row * illegal_move_penalty

        if (random.random() < exploration_rate):
            # Return a random state
            # Generate a random tensor
            # Mask out illegal moves
            # return the max
            rand_tensor = torch.rand(self.action_states)

            # Use absolute value so that all filled spaces are 1
            masked_rand = rand_tensor + illegal_move_mask

            return torch.argmax(masked_rand).view(1,1)
        else:
            # Return a state based on the policy
            q_s = self.policy_net(state.unsqueeze(0))
            # Mask out illegal moves (any column with a 1 in the first space)
            # Use absolute value so that all filled spaces are 1
            masked_qs = q_s + illegal_move_mask

            return torch.argmax(masked_qs).view(1,1)

    def experience_replay(self):
        if (self.batch_size > len(self.memory)):
            return 0

        batch_loss = []

        # Get a batch of samples from the replay membor
        experience_batch = self.memory.sample(self.batch_size)

        # Convert from a list of tuples to a tuple of lists
        transitions = ExperienceTuple(*zip(*experience_batch))

        # Put the transitions into a batch that pytorch undestands
        if (self.conv_model):
            state_batch = torch.stack(transitions.state)
        else:
            state_batch = torch.vstack(transitions.state)

        
        action_batch = torch.cat(transitions.action)

        if (self.conv_model):
            next_state_batch = torch.stack(transitions.next_state)
        else:
            next_state_batch = torch.vstack(transitions.next_state)

        reward_batch = torch.cat(transitions.reward)
        
        # Convert to numbers to be used as a mask
        done_batch = torch.cat(transitions.done).long()

        # Prediction is based on the target net
        # (if done then the future reward is 0 )
        targets = self.target_net(next_state_batch)
        # Get the max for each element of the batch

        max_targets = targets.max(dim = 1).values

        predicted = reward_batch + (self.gamma * max_targets)#torch.mul((self.gamma * max_targets), done_batch)
        # expected is based on the policy net
        expected = self.policy_net(state_batch).gather(1, action_batch).squeeze()

        self.optimizer.zero_grad()

        loss = self.loss(predicted, expected)
        loss.backward()
        batch_loss.append(int(loss))

        self.optimizer.step()
        self.scheduler.step()

        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        
        self.epsilon = max(self.epsilon * (1 - self.epsilon_decay), self.min_epsilon)

        return sum(batch_loss)/len(batch_loss)


    def train(self, episodes):
        eps_losses = []
        avg_ep_rewards = []
        eps_rewards = []
        # ep_rewards = []

        last_time = time.time()
        for ep_num in range(episodes):
            
            done = False
            # Reset the training environment
            # Get the initial state from the environment
            state = self.env.reset()
            if (self.conv_model):
                # Convert 1d input to 2d
                state = state.view(1, self.board_height, self.board_width)

            # Run through the game until it ends
            ep_loss = 0
            total_rewards = []
            while not done:

                action = self.predict(state)
                next_state, reward, done, info = self.env.step(action)

                total_rewards.append(reward)

                if (self.conv_model):
                    # Convert 1d input to 2d
                    next_state = next_state.view(1, self.board_height, self.board_width)

                self.memory.store(state, action, next_state, torch.tensor([reward]), torch.tensor([done]))
                # Since connect 4 is horizontally symetric we can "cheat" and add the reflection of the board aswell
                if (self.conv_model):
                    reflected_next_state = torch.fliplr(next_state[0]).view(1, self.board_height, self.board_width)
                    self.memory.store(state, action, reflected_next_state, torch.tensor([reward]), torch.tensor([done]))
                state = next_state

            # optimize the model
            batch_loss = self.experience_replay()
            # ep_loss += batch_loss

            # print(sum(total_rewards))
            # np.mean(total_rewards)
            eps_rewards.append(sum(total_rewards))

            if ep_num % self.target_update_rate == 0:
                # Update the target net to use the trained policy net
                # eps_rewards.append(sum(ep_rewards)/len(ep_rewards))
                # ep_rewards = []
                avg_reward_over_ep = np.mean(eps_rewards)
                avg_ep_rewards.append(avg_reward_over_ep)
                eps_rewards = []
                self.target_net.load_state_dict(self.policy_net.state_dict())
        
                curr_time = time.time()
                print(f'Episode {ep_num} Complete, Runtime {round(curr_time - last_time, 2)}s, Avg Reward = {round(avg_reward_over_ep, 2)}, Epsilon = {round(self.epsilon, 2)}', end='\r')
                last_time = curr_time
        return avg_ep_rewards[1:]


if __name__ == '__main__':
    height = 6
    width = 7
    x = 4
    state_size = height * width
    action_size = width
    board = ConnectXBoard(height=height, width=width, x=x)
    env = ConnectXGym(board, RandomConnect4Player(), 1, -1)
    agent = DQNAgent(env, board_height=height, board_width=width, action_states=action_size, batch_size=50, epsilon=.4, gamma=.9, lr=0.01)
    agent.train(50)
