import numpy as np

def find_nearest(array, values):
    ''' Find the nearest elements of 'array' to 'values' '''
    return np.searchsorted(array[:-1], values, side="left")