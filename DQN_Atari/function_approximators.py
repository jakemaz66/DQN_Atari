import torch
import torch.nn as nn
import torch.autograd as autograd 
import torch.nn.functional as F
import random
import random

class FunctionApproximator(nn.Module):

    def __init__(self, obervation_size, hidden_size, action_size):
        """This initializes a PyTorch Neural Net that acts as an approximator
           for the value function Q

        Args:
            obervation_size (_type_): the size of the observation space
            hidden_size (_type_): the number of hidden layers in the network
            action_size (_type_): the number of possible actions
        """
        #Calling parent constructor
        super(FunctionApproximator, self).__init__()

        self.layer1 = nn.Linear(obervation_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, action_size)

    def forward(self, observation):
        observation = F.relu(self.layer1(observation))
        observation = F.relu(self.layer2(observation))
        return self.layer3(observation)
    

class AtariFunctionApproximator(nn.Module):
    """This is a CNN to play Atari Games"""

    def __init__(self, frame, action_size):
        #Calling parent constructor
        super(AtariFunctionApproximator, self).__init__()

        self.frame = frame
        self.action_size = action_size

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
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.action_size)

        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.feed_forward(x)
        return x

    def feature_size(self):
        #Self.frame is a tuple of the frame size (4, 84, 84)
        return self.features(autograd.Variable(torch.zeros(1,*self.frame))).view(1, -1).size(1)
    
    def act(self, state, epsilon, env):
        #Epsilon greedy exploration
        if random.random() > epsilon:
            q_value = self.forward(state.unsqueeze(0).to(torch.float32))
            #Act according to highest q value, getting the index of the action
            action  = q_value.max(1)[1].data[0]
        else:
            #Take random action
            action = random.randrange(env.action_space.n)

        return torch.tensor([action])
    


    

