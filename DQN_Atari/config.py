"""This contains all the hyperparameters for the training"""

#number of episodes to sample from memory
BATCHSIZE = 128

#learning rate for optimizer
LR = 1e-4

#decay rate for rewards
GAMMA = 0.99

#decay rate for e-greedy exploring
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000

#soft update paramter for the target network
SOFTUPDATE = 0.005

#number of episodes to train for
NUMEPISODES = 5000