import time
import torch
import numpy as np


class GeneticAgent:
    def __init__(self, env, population=10):
        self.env = env
        self.population = population
        self._reset()

    def _reset(self):
        return 0

    def train(self):
        return 0
