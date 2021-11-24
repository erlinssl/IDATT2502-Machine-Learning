import numpy as np


def t_and_p(array):
    print(len(array))
    array = [*zip(*array)]
    print(len(array))
    print(array, "\n")


i_array = [[1, 1, 1, 1]]
t_and_p(i_array)

o_array = [[1, 1],
           [1, 1]]
t_and_p(o_array)

s_array = [[0, 1, 1],
           [1, 1, 0]]
t_and_p(s_array)
