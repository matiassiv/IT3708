from math import sqrt
from numba import jit


def rgb_euclidean(arr, p_1, p_2):
    d0 = arr[p_1[0], p_1[1], 0] - arr[p_2[0], p_2[1], 0]
    d1 = arr[p_1[0], p_1[1], 1] - arr[p_2[0], p_2[1], 1]
    d2 = arr[p_1[0], p_1[1], 2] - arr[p_2[0], p_2[1], 2]

    return sqrt(d0**2 + d1**2 + d2**2)