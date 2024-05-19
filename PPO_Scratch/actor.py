import torch
import torch.nn as nn
import torch.autograd as autograd 
import torch.nn.functional as F
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
    

class PongActor(nn.Module):

    def __init__(self, frame, action_size, hidden_size=512):
        """This initializes a PyTorch Neural Net that acts as an approximator
           for the value function Q

        Args:
            obervation_size (_type_): the size of the observation space
            hidden_size (_type_): the number of hidden layers in the network
            action_size (_type_): the number of possible actions
        """
        #Calling parent constructor
        super(PongActor, self).__init__()

        self.frame = frame

        self.features = nn.Sequential(
            nn.Conv2d(self.frame[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()     
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(self.feature_size(), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)

        )

    def feature_size(self):
        #Self.frame is a tuple of the frame size (4, 84, 84)
        return self.features(autograd.Variable(torch.zeros(1,*self.frame))).view(1, -1).size(1)

    def forward(self, observation):
        if observation.dim() == 3:
            observation = observation.unsqueeze(0)
        features = self.features(observation)
        batch_dim = features.size(0)
        features = features.view(batch_dim, -1)

        policy_logits = self.feed_forward(features)
        policy = F.softmax(policy_logits, dim=-1)
        policy = Categorical(policy)

        return policy


    

