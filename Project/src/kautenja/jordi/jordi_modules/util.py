import gym_tetris
import numpy as np
from gym_tetris.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from processing import wrap_env

"""
The heuristics methods assumes a valid state, meaning there is no
floating current_piece, and are to be called after a potential
piece placement has been simulated.
"""


def _get_tops(state):
    tops = []
    for x in range(len(state[0])):
        for y in range(len(state)):
            if state[y][x] == 0:
                continue
            tops.append(y)
            break
    return tops


def get_holes(state):
    holes = 0
    for x in range(len(state[0])):
        first = -1
        for y in range(len(state)):
            if state[y][x] == 0 and first < 0:
                continue
            if first < 0:
                first = y
            if state[y][x] == 0:
                holes += 1
    return holes


def get_clears(state):
    clears = 0
    for row in range(len(state)):
        if np.all(state[row] == 1):
            clears += 1
    return clears


def get_bumpiness(state):
    tops = _get_tops(state)
    return sum(np.abs(np.diff(tops)))


def get_aggregate_height(state):
    tops = _get_tops(state)
    return sum([len(state) - y for y in tops])


def get_height_diff(state):
    tops = _get_tops(state)
    return max(tops) - min(tops)


def get_heuristics(state):
    return get_holes(state), get_clears(state), get_bumpiness(state)  # , self._aggregate_height(state)


def main():
    # For testing purposes
    state = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
    # print(state)

    print("Tops", _get_tops(state))
    print("Holes", get_holes(state))
    print("Clears", get_clears(state))
    print("Bumpiness", get_bumpiness(state))
    print("Aggregate", get_aggregate_height(state))
    print("Height diff", get_height_diff(state))

    # print("Heuristics", get_heuristics(state))


if __name__ == "__main__":
    main()
