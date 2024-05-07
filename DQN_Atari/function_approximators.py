import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class FunctionApproximator(nn.Module):

    def __init__(self, obervation_size, hidden_size, action_size):
        """This initializes a PyTorch Neural Net that acts as an approximator
           for the value function Q

        Args:
            obervation_size (_type_): the size of the observation space
            hidden_size (_type_): the number of hidden layers in the network
            action_size (_type_): the number of possible actions
        """

        self.linear1 = nn.Linear(obervation_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, action_size)

    def forward(self, observation):
        observation = F.relu(self.linear1(observation))
        observation = F.relu(self.linear2(observation))

        return observation