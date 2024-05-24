from collections import deque, namedtuple
import numpy as np
import random 


"""EnvStep is a transition from one state to the next. It remembers the current state of agent,
   the action selected, the observation from the environment, and the reward
"""
EnvStep = namedtuple('EnvStep', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(EnvStep(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class PrioritizedMemoryReplay:
    """This class acts as the memory buffer for the network to learn on"""

    def __init__(self, max_len, prob_alpha=0.6):
        self.max_len = max_len
        self.memory = deque(maxlen=self.max_len)
        #A numpy array that stores the priorities of transitions
        self.priority = np.zeros(max_len)
        self.prob_alpha = prob_alpha
        self.pos = 0

    def sample(self, batch_size, beta=0.4):
        """We need to sample episodes randomly from the memory buffer"""
        if self.priority[-1] != 0:
            prios = self.priority
        else:
            prios = self.priority[:self.pos]
        #Probability of sampling a transition is the priority raise to alpha / sum of all priorities raised to alpha
        probs  = prios ** self.prob_alpha

        if (probs[:] != 0).sum() != 0:
            probs /= probs.sum()

        #Sampling from self.memory with probabilities of the priorities
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        #Importance sampling weight
        total = len(self.memory)
        weights  = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)

        return samples, indices, weights
    
    def update_priorities(self, batch_indices, batch_priorities):
        """This function updates the priorities to give higher priority to transitions
           with a greater TD Error
        """
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priority[idx] = prio
    
    def add_episode(self, *args):
        """This function appends an episode to our memory"""

        #The maximum priority transition
        #When we add an episode, it automatically gets the highest priority because we don't know
        #TD Error, this ensures all transitions are most likely to be seen at least once
        max_priority = self.priority.max() if self.memory else 1.0

        self.memory.append(EnvStep(*args))

        self.priority[self.pos] = max_priority

        #Update position after adding each episode
        self.pos = (self.pos + 1) % self.max_len 

    def length(self):
        return len(self.memory)

