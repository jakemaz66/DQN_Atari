"""This contains all the hyperparameters for the training"""

#number of episodes to sample from memory
BATCHSIZE = 32

#learning rate for optimizer
LR = 0.0001

#decay rate for rewards
GAMMA = 0.99

GAE_LAMBDA = 0.95

#decay for clip
EPSILON = 0.1

#number of episodes to train for
NUMEPISODES = 500

#Number of steps to take for each PPO trajectory update
TRAJLENGTH = 128

OPTM_EPOCHS = 3

ENTROPY_TERM = 0.01
