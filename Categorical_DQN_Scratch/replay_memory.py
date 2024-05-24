from collections import namedtuple, deque
import numpy as np
import random

Memory = namedtuple("batch", ("state", "action", "reward", "next_state", "done"))

class MemoryReplay:

    def __init__(self, max_len):
        self.memory = deque(maxlen=max_len)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def remember(self, state, action, reward, next_state, done):
        memory = Memory(state, action, reward, next_state, done)
        self.memory.append(memory)

    def length(self):
        return len(self.memory)