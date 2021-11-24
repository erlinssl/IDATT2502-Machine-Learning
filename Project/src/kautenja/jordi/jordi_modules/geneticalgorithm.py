import time

import gym_tetris
import torch
import numpy as np
import random
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import SIMPLE_MOVEMENT
from .processing import wrap_env
from . import tetris_util as utils


class GenePool:
    def __init__(self, population=100, mutateChance=0.05, games=5, moves=150, replacePercent=0.3):
        self.population = population
        self.mutateChance = mutateChance
        self.maxGames = games
        self.maxMoves = moves
        self.replacePercent = 0.3

    def train(self, generations):
        print("training")
        return 0

    def train_test(self):
        print("testing")
        candidate = GeneticAgent(random.random(), random.random(), random.random())
        env = gym_tetris.make('TetrisA-v1')
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = wrap_env(env, buffersize=1)
        env.seed(4)
        state = env.reset()
        state, _, _, info = env.step(0)
        best = candidate.best_move(state, info['current_piece'])
        print(best)


class GeneticAgent:
    def __init__(self, top_weight, hole_weight, clear_weight):
        self.hole_weight = hole_weight
        self.clear_weight = clear_weight
        self.bump_weight = top_weight

    def best_move(self, state, current_piece):
        global test_state
        rots, shape = utils.get_rotations(current_piece)
        best_rotation, best_x_offset, y_steps_best, best_score = None, None, None, None

        for rot in range(rots):
            current_shape = np.rot90(shape, - rot)
            for x in range(len(state[0][0]) - len(current_shape[0])):
                y, new_state = utils.y_collision_state(state[0], current_piece, current_shape, x)
                # y, new_state = utils.y_collision_state(use_state, current_shape, x)
                # print(current_shape)
                # print("x_off =", x, "collision after", y, "\n")  # Debugging

                score = self._calc_score(new_state, current_piece)
                if best_score is None or score > best_score:
                    best_rotation = rot
                    best_x_offset = x
                    y_steps_best = y
                    best_score = score

        # TODO optimize actions
        actions = [[1] * best_rotation,
                   [4] * 11,  # Very primitive solution, should be optimized
                   [4] * best_x_offset,
                   [5] * y_steps_best]
        return actions

    def _calc_score(self, new_state, current_piece):
        holes, clears, bumpiness = utils.get_heuristics(new_state, current_piece)
        return self.clear_weight * clears - self.hole_weight * holes - self.bump_weight * bumpiness


test_state = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                       [0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
                       [1, 1, 1, 0, 1, 1, 0, 1, 1, 0],
                       [0, 1, 1, 1, 1, 1, 0, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                       ])

if __name__ == "__main__":
    print("")
