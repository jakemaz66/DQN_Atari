from collections import deque, namedtuple
import random


"""EnvStep is a transition from one state to the next. It remembers the current state of agent,
   the action selected, the observation from the environment, and the reward
"""
EnvStep = namedtuple('EnvStep', ['state', 'action', 'next_state', 'reward'])


class MemoryReplay:
    """This class acts as the memory buffer for the network to learn on"""

    def __init__(self, max_len):
        self.memory = deque(maxlen=max_len)

    def sample(self, batch_size):
        """We need to sample episodes randomly from the memory buffer"""
        return random.sample(self.memory, batch_size)
    
    def add_episode(self, *args):
        """This function appends an episode to our memory"""
        self.memory.append(EnvStep(*args))

