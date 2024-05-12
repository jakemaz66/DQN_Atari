"""This contains all the hyperparameters for the training"""

#number of episodes to sample from memory
BATCHSIZE = 32

#learning rate for optimizer
LR = 1e-4

#decay rate for rewards
GAMMA = 0.95

#decay rate for e-greedy exploring
EPSILON_DECAY = .9/100000
EPSILON_MIN = .05

#soft update paramter for the target network
SOFTUPDATE = 0.005

#number of episodes to train for
NUMEPISODES = 5000

#number of frames for Atari
NUMFRAMES = 100_000