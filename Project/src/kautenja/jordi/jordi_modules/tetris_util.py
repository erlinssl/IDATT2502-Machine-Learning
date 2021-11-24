import time
import numpy as np

EXT_PIECE_DICT = {'Td': 0, 'Tr': 1, 'Tu': 2, 'Tl': 3,
                  'Jd': 4, 'Jr': 5, 'Ju': 6, 'Jl': 7,
                  'Zh': 8, 'Zv': 9,
                  'O': 10,
                  'Sh': 11, 'Sv': 12,
                  'Ld': 13, 'Lr': 14, 'Lu': 15, 'Ll': 16,
                  'Ih': 17, 'Iv': 18, None: 19}

PIECE_DICT = {'T': 0, 'J': 1, 'Z': 2, 'O': 3, 'S': 4, 'L': 5, 'I': 6, None: 7}

"""
The heuristics methods assumes a valid state, meaning there is no
floating current_piece, and are to be called after a potential
piece placement has been simulated.
If this is not the case, _trim_state() used first, which is
a somewhat primitive method. It was the simplest implementation
that I could think to implement in the remaining timeframe.
"""


def _get_tops(state):
    tops = []
    for x in range(len(state[0])):
        flag = True
        for y in range(len(state)):
            if state[y][x] == 0:
                continue
            tops.append(y)
            flag = False
            break
        if flag:
            tops.append(20)  # if column had no collisions, top is bottom
    return tops


def _get_holes_old(state):
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


def get_holes(state):
    tops = _get_tops(state)
    holes = 0
    for x in range(len(state[0])):
        for y in range(tops[x], len(state)):
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


def get_heuristics(state, current_piece=None):
    """
    The method thats generally called from the outside to get heuristics.
    Takes one layer of the state, as well as whichever piece is currently
    in play, optionally.
    First makes sure that the current piece is not counted when calculating
    the heuristics by calling _trim_state().
    """
    state = _trim_state(state, current_piece)
    return get_holes(state), get_clears(state), get_bumpiness(state)


def _trim_state(state, current_piece):
    """
    Used to remove the current_piece shape from the state image.
    Operates on the idea that even if the state had something in
    the cells that are deleted, that would mean the game has ended
    anyways. This little 'oversight' shouldn't mean too much in
    the grand scheme of things.
    """
    if current_piece is None:
        return state
    elif current_piece == 'Td':
        state[0][4:7] = 0
        state[1][5] = 0
    elif current_piece == 'Jd':
        state[0][4:7] = 0
        state[1][6] = 0
    elif current_piece == 'Zh':
        state[0][4:6] = 0
        state[1][5:7] = 0
    elif current_piece == 'O':
        state[0][4:6] = 0
        state[1][4:6] = 0
    elif current_piece == 'Sh':
        state[0][5:7] = 0
        state[1][4:6] = 0
    elif current_piece == 'Ld':
        state[0][4:7] = 0
        state[1][4] = 0
    elif current_piece == 'Ih':
        state[0][3:7] = 0
    else:
        print("UNEXPECTED PIECE WHILE TRIMMING", current_piece)
        time.sleep(10)
    return state


def get_rotations(current_piece):
    """
    Returns the number of given tetromino's unique rotations and
    its base shape.
    """
    if current_piece is None:
        return None
    elif current_piece == 'Td':
        return (4, [[1, 1, 1],
                    [0, 1, 0]])
    elif current_piece == 'Jd':
        return (4, [[1, 1, 1],
                    [0, 0, 1]])
    elif current_piece == 'Zh':
        return (2, [[1, 1, 0],
                    [0, 1, 1]])
    elif current_piece == 'O':
        return (1, [[1, 1],
                    [1, 1]])
    elif current_piece == 'Sh':
        return (2, [[0, 1, 1],
                    [1, 1, 0]])
    elif current_piece == 'Ld':
        return (4, [[1, 1, 1],
                    [1, 0, 0]])
    elif current_piece == 'Ih':
        return 2, [[1, 1, 1, 1]]
    else:
        print("UNEXPECTED PIECE GETTING ROTS:", current_piece)
        time.sleep(10)
    return None


def y_collision_state(state, current_piece, shape, x_offset):
    """
    Used to 'simulate' a tetris environment so we can find out how
    long it takes for the given piece (shape) to collide with
    the blocks placed on the board (state).
    Also draws the new state that the said collision would
    result in.
    """
    coords = []

    state = _trim_state(state, current_piece)

    new_state = np.copy(state)

    for y in range(len(shape)):
        for x in range(len(shape[0])):
            if shape[y][x] != 0:
                coords.append((y, x))

    collisions = []
    x_coords = []
    for y, x in coords:
        steps = 0
        flag = True
        while y < len(new_state)-1:
            if new_state[y + 1][x + x_offset] != 0:
                collisions.append(steps)
                x_coords.append(x+x_offset)
                flag = False
                break
            steps += 1
            y += 1
        if flag:
            collisions.append(steps)
            x_coords.append(x+x_offset)

    y = min(collisions)

    for row in range(len(shape)):
        for column in range(len(shape[0])):
            if shape[row, column] != 0:
                new_state[y + row][column + x_offset] = 1

    # print(new_state)
    return y, new_state  # TODO May need to return y-1 depending on env


def _time(method, state, iter_n):
    """
    Debugging method used to check a methods performance
    """
    start = time.perf_counter()
    for _ in range(iter_n):
        method(state)
    end = time.perf_counter()
    return end - start


def main():
    # For testing purposes
    state = np.array([[0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
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

    print(state.shape)

    print("Tops", _get_tops(state), "\n")
    print("Holes", get_holes(state))
    print("Clears", get_clears(state))
    print("Bumpiness", get_bumpiness(state))
    print("Aggregate", get_aggregate_height(state))
    print("Height diff", get_height_diff(state))

    print("\nHeuristics", get_heuristics(state, 'O'))


if __name__ == "__main__":
    main()

