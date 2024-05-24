import torch
from torch import nn
import torch.nn.functional as F

class CategoricalDQN(nn.Module):

    def __init__(self, n_input, n_output, n_atoms, Vmin, Vmax, hidden_size=128):
        super(CategoricalDQN, self).__init__()

        self.n_atoms = n_atoms
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.n_input = n_input
        self.n_output = n_output


        self.model = nn.Sequential(
            nn.Linear(n_input, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            #Output is the distribution over #atoms for each action
            nn.Linear(hidden_size, n_output * n_atoms)
        )

    def forward(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32)

        #Pytorch expects a batch dimension, added 1 with unsqueeze()
        if obs.shape != (32, 8):
            obs = self.model(obs).unsqueeze(0)
            obs = F.softmax(obs.view(-1, self.n_atoms)).unsqueeze(0)
        else:
            obs = self.model(obs)
            obs = F.softmax(obs.unsqueeze(1).view(-1, self.n_output, self.n_atoms))

        #Output will be of size 1 x #actions x #atoms
        return obs
    
    def act(self, state):
        '''Retrieve the action with the maximum return'''
        dist = self.forward(state)

        #projecting dist onto support 
        dist = dist * torch.linspace(self.Vmin, self.Vmax, self.n_atoms)

        #Max(1) gives index rather than value with max()
        action = dist.sum(-1).max(1)[1].item()

        return action


    


