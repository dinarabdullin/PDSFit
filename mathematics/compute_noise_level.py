import numpy as np


def compute_noise_level(y_im):
    ''' Computes the noise level of the PDS time trace '''
    size_y_im = len(y_im)
    size_noise_std = int(2/3 * float(size_y_im))
    noise_std = np.std(y_im[-size_noise_std:])
    return noise_std