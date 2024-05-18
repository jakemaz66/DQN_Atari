import torch
import torch.nn as nn
import torch.autograd as autograd 
import torch.nn.functional as F
import random
import random
import config
from torch.distributions import Categorical

class Actor(nn.Module):

    def __init__(self, obervation_size, action_size, hidden_size=64):
        """This initializes a PyTorch Neural Net that acts as an approximator
           for the value function Q

        Args:
            obervation_size (_type_): the size of the observation space
            hidden_size (_type_): the number of hidden layers in the network
            action_size (_type_): the number of possible actions
        """
        #Calling parent constructor
        super(Actor, self).__init__()

        self.layer1 = nn.Linear(obervation_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, action_size)

    def forward(self, observation):
        observation = F.tanh(self.layer1(observation))
        observation = F.tanh(self.layer2(observation))
        observation = F.tanh(self.layer3(observation))
        return Categorical(F.softmax(self.layer4(observation)))


    

