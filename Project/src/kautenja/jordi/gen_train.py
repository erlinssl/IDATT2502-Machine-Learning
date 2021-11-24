import time
import random
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import gym_tetris
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import SIMPLE_MOVEMENT
from jordi_modules.processing import wrap_env
from jordi_modules.tetris_util import get_heuristics
from jordi_modules.geneticalgorithm import GenePool, GeneticAgent


def main():
    # gene_pool = GenePool()
    # strongest = gene_pool.train_test()

    env = gym_tetris.make('TetrisA-v1')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = wrap_env(env, buffersize=1)

    agent = GeneticAgent(random.random(), random.random(), random.random())

    state = env.reset()
    state, reward, done, info = env.step(0)

    step = 0
    while not done:

        step += 1
        env.render()

        if info['current_piece'] is None:
            env.step(0)

        actions = agent.best_move(state, info['current_piece'])
        print(info['current_piece'])

        for sub_arr in actions:
            for action in sub_arr:
                state, reward, done, info = env.step(action)

    time.sleep(5)


if __name__ == "__main__":
    main()
