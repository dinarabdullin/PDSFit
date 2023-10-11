import numpy as np


def find_nearest(arr, val):
    """Find the nearest elements of 'arr' to 'val'"""
    return np.searchsorted(arr[:-1], val, side = "left")