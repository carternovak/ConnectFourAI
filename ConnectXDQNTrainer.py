from collections import deque, namedtuple
import random
import torch
from torch import nn

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemeory:
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, state_size, action_size, layer_sizes=[]):
        super(DQN, self).__init__()
        computed_layer_sizes = [state_size] + layer_sizes + [action_size]
        self.layers = torch.nn.ModuleList(
                                    [nn.Linear(computed_layer_sizes(index), computed_layer_sizes(index + 1)) for index in range(len(computed_layer_sizes)+1)]
                                    )
        self.relu = nn.ReLU()

    def forward(self, x):
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if (index < len(self.layers) - 1):
                x = self.relu(x)
        return x