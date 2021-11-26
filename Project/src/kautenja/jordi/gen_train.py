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
    # agent = gene_pool.train(generations=10)

    agent = GeneticAgent  # Random weights
    # agent = GeneticAgent(0.9521941443319699, 0.8479778551751184, 0.3053698551679609)  # weights from training

    env = gym_tetris.make('TetrisA-v1')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = wrap_env(env, buffersize=1, skip=0, heuristic=True)

    rewards = []
    for _ in range(10):
        agent.random()
        state = env.reset()
        total_reward = 0
        state, reward, done, info = env.step(0)
        step = 0
        while not done and 1:
            step += 1
            env.render()

            if state[0][0][5] != 0:  # TODO Find a better way to find out if piece has been placed
                actions = []
                actions = agent.best_move(state, info['current_piece'])

                for sub_arr in actions:
                    for action in sub_arr:
                        env.render()
                        if done:
                            break
                        state, reward, done, info = env.step(action)
                        total_reward += reward
                        if done:
                            break
                        state, reward, done, info = env.step(0)  # Buffers an extra action to hinder lossy inputs
                        total_reward += reward
            else:
                state, reward, done, info = env.step(5)
                total_reward += reward

            if info['current_piece'] is None:
                env.step(0)
        print(agent.get_weights())
        rewards.append(total_reward)

    plt.plot(rewards)
    plt.title("Rewards for random heuristic agents")
    plt.xlabel("Agent #")
    plt.ylabel("Lines cleared")
    plt.show()


if __name__ == "__main__":
    main()
