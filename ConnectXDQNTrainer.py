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

    def __init__(self, input_size, action_states, dropout = None) -> None:
        super(DQN, self).__init__()
        self.l1 = nn.Linear(input_size, 35)
        self.l2 = nn.Linear(35, 30)
        self.l3 = nn.Linear(30, 25)
        self.l4 = nn.Linear(25, 20)
        self.l5 = nn.Linear(20, 15)
        self.l6 = nn.Linear(15, action_states)
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

# Pseudo Code
# For episode in episodes
class DQNAgent:
    def __init__(self, env, board_width, board_height, action_states, batch_size, lr, gamma, epsilon, epsilon_decay = 0.01, min_epsilon=0.1, target_update_rate = 100, pre_trained_target = None, pre_trained_policy = None) -> None:
        self.memory = ReplayMemory(50000)
        self.env = env
        self.board_width = board_width
        self.board_height = board_height
        input_size = board_width * board_height
        self.input_size = input_size
        self.action_states = action_states
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
            # Generate a random tensor
            # Mask out illegal moves
            # return the max
            rand_tensor = torch.rand(self.action_states)
            illegal_move_penalty = -9999999
            #print(state[-self.board_width:])
            # Use absolute value so that all filled spaces are 1
            illegal_move_mask = torch.abs(state[-self.board_width:]) * illegal_move_penalty
            #print(illegal_move_mask)
            masked_rand = rand_tensor + illegal_move_mask
            return torch.argmax(masked_rand).view(1,1)
            #return torch.tensor(random.randrange(self.action_states)).view(1,1)
        else:
            # Return a state based on the policy
            q_s = self.policy_net(state)
            # Mask out illegal moves (any column with a 1 in the first space)
            illegal_move_penalty = -9999999
            #print(state[-self.board_width:])
            # Use absolute value so that all filled spaces are 1
            illegal_move_mask = torch.abs(state[-self.board_width:]) * illegal_move_penalty
            #print(illegal_move_mask)
            masked_qs = q_s + illegal_move_mask
            return torch.argmax(masked_qs).view(1,1)

    def experience_replay(self):
        if (self.batch_size > len(self.memory)):
            return 0

        batch_loss = []
        experience_batch = self.memory.sample(self.batch_size)

        transitions = ExperienceTuple(*zip(*experience_batch))
        # print(transitions.state[0].shape)
        # print("state")
        # print(transitions.state)
        # print("action")
        # print(transitions.action)
        state_batch = torch.vstack(transitions.state)
        # print(state_batch.shape)
        action_batch = torch.cat(transitions.action)
        next_state_batch = torch.vstack(transitions.next_state)
        reward_batch = torch.cat(transitions.reward)
        # Convert to numbers to be used as a mask
        done_batch = torch.cat(transitions.done).long()
        # print('state')
        # print(state_batch.shape)
        # print('action')
        # print(action_batch.shape)
        # print('next state')
        # print(next_state_batch.shape)
        # print('reward')
        # print(reward_batch.shape)
        # print('done')
        # print(done_batch.shape)

        # Prediction is based on the target net
        # (if done then the future reward is 0 )
        targets = self.target_net(next_state_batch)
        # Get the max for each element of the batch
        max_targets = targets.max(dim=1).values
        predicted = reward_batch + torch.mul((self.gamma * max_targets), done_batch)
        # expected is based on the policy net
        expected = self.policy_net(state_batch).gather(1, action_batch).squeeze()
        # print('predicted')
        # print(predicted.shape)
        # print('expected')
        # print(expected.shape)
        # print("predicted")
        # print(predicted.shape)
        # print("expected")
        # print(expected.shape)
        #print("predicted")
        #print(predicted)
        #print("expected")
        #print(expected)
        loss = self.loss(predicted, expected)
        loss.backward()
        batch_loss.append(int(loss))
        # print(int(loss))
        self.optimizer.step()

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        # for experience in experience_batch:
        #     # Prediction is based on the target net
        #     # (if done then the future reward is 0 )
        #     predicted = torch.tensor(experience.reward) + ((self.gamma * self.target_net(experience.next_state).max()) if not experience.done else 0)
        #     # expected is based on the polic net
        #     expected = self.policy_net(experience.state)[experience.action][0]
        #     #print("predicted")
        #     #print(predicted)
        #     #print("expected")
        #     #print(expected)
        #     loss = self.loss(predicted, expected)
        #     loss.backward()
        #     # print(int(loss))
        #     self.optimizer.step()
        
        self.epsilon = max(self.epsilon * (1 - self.epsilon_decay), self.min_epsilon)

        # return 0
        return sum(batch_loss)/len(batch_loss)


    def train(self, episodes):
        eps_losses = []
        eps_rewards = []
        ep_rewards = []
        for ep_num in range(episodes):
            done = False
            # Reset the training environment
            # Get the initial state from the environment
            state = self.env.reset()
            # Run through the game until it ends
            ep_loss = 0
            while not done:

                action = self.predict(state)
                next_state, reward, done, info = self.env.step(action)
                ep_rewards.append(reward)
                self.memory.store(state, action, next_state, torch.tensor([reward]), torch.tensor([done]))
                state = next_state
                # optimize the model
                batch_loss = self.experience_replay()
                ep_loss += batch_loss

            eps_losses.append(ep_loss)
            if ep_num % self.target_update_rate == 0:
                # Update the target net to use the trained policy net
                print(f'Episode {ep_num + 1} Done, Avg. Reward = {sum(ep_rewards)/len(ep_rewards)}')
                eps_rewards.append(sum(ep_rewards)/len(ep_rewards))
                ep_rewards = []
                self.target_net.load_state_dict(self.policy_net.state_dict())
        return eps_rewards


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
