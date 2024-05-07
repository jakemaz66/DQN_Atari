"""This contains all the hyperparameters for the training"""

#number of episodes to sample from memory
BATCHSIZE = 128

#learning rate for optimizer
LR = 0.001

#decay rate for rewards
GAMMA = 0.99

#decay rate for e-greedy exploring
EPSILON = 0.05
EXPLOREDECAY = 0.9

#soft update paramter for the target network
SOFTUPDATE = 0.05

#number of episodes to train for
NUMEPISODES = 50