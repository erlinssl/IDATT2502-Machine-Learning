import time

import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt
import gym_tetris
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import SIMPLE_MOVEMENT
from jordi_modules.processing import wrap_env
from jordi_modules.tetris_util import get_heuristics


def main():
    env = gym_tetris.make('TetrisA-v1')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = wrap_env(env, buffersize=1)

    state = env.reset()
    state, reward, done, info = env.step(0)

    step = 0
    while not done:

        step += 1
        env.render()

        if info['current_piece'] is None:
            env.step(0)

        if state[0][0][5] == 1:
            print(get_heuristics(state[0], info['current_piece']))

        state, reward, done, info = env.step(0)

    time.sleep(5)


if __name__ == "__main__":
    main()
