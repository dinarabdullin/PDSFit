import numpy as np


def chi2(x1, x2, noise_std=0):
    ''' chi2 '''
    if noise_std:
        return np.sum((x1 - x2)**2)  / noise_std**2
    else:
        return np.sum((x1 - x2)**2)