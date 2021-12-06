import numpy as np

def find_nearest(array, values):
    return np.searchsorted(array[:-1], values, side="left")