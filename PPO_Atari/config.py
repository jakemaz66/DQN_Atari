"""This contains all the hyperparameters for the training"""

#number of episodes to sample from memory
BATCHSIZE = 5

#learning rate for optimizer
LR = 0.00025

#decay rate for rewards
GAMMA = 0.99

GAE_LAMBDA = 0.95

#decay for clip
EPSILON = 0.2

#number of episodes to train for
NUMEPISODES = 500

#Number of steps to take for each PPO trajectory update
TRAJLENGTH = 20

OPTM_EPOCHS = 3
