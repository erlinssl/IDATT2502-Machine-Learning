import random
from collections import deque


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def append(self, transition):
        if len(self.memory) == self.capacity:
            del self.memory[0]
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)  # DIFF

    def __len__(self):
        return len(self.memory)
