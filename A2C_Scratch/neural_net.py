from torch import nn
import torch.nn.functional as F


class NeuralNet(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_size=256):
        super(NeuralNet, self).__init__()

        self.linear_critic1 = nn.Linear(input_dims, hidden_size)
        self.linear_critic2 = nn.Linear(hidden_size, 1)

        self.linear_actor1 = nn.Linear(input_dims, hidden_size)
        self.linear_actor2 = nn.Linear(hidden_size, output_dims)

    def forward(self, x):
        state_value = F.relu(self.linear_critic1(x))
        state_value = F.relu(self.linear_critic2(state_value))

        policy = F.relu(self.linear_actor1(x))
        policy = F.softmax(self.linear_actor2(policy))

        #Returning a categorical distribution over all possible actions
        return state_value, policy

