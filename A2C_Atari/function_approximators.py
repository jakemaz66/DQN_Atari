import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class FunctionApproximator(nn.Module):

    def __init__(self, obervation_size, action_size, hidden_size=256):
        """This initializes a PyTorch Neural Net that acts as an approximator
           for the value function Q

        Args:
            obervation_size (_type_): the size of the observation space
            hidden_size (_type_): the number of hidden layers in the network
            action_size (_type_): the number of possible actions
        """
        #Calling parent constructor
        super(FunctionApproximator, self).__init__()

        self.critic_linear1 = nn.Linear(obervation_size, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(obervation_size, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, action_size)

    def forward(self, observation):
        observation = torch.tensor(observation)

        value = F.relu(self.critic_linear1(observation))
        value = self.critic_linear2(value)
        
        policy_dist = F.relu(self.actor_linear1(observation))
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=0)

        return value, policy_dist

