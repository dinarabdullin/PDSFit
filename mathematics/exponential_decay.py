import numpy as np


def exponential_decay(x, A, k, p):
    ''' Exponential decay '''
    arg = k * np.abs(x)**(p)
    return A * np.exp(-1.0 * np.where((arg > 1e-10), arg, 0.0))