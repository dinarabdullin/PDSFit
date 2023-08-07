import numpy as np


def ceil_with_precision(a, precision=0):
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)


def floor_with_precision(a, precision=0):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)