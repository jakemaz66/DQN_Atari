"""This contains all the hyperparameters for the training"""

#learning rate for agent
LR = 1e-4

#decay rate for rewards
GAMMA = 0.99

#decay rate for e-greedy exploring
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000

#number of episodes to train for
NUMEPISODES = 15000

NUMSTEPS = 300
