'''
A modified histogram with "bins" being bin centers instead of bin edges
'''

import numpy as np

def histogram(a, bins=10, range=None, normed=None, weights=None, density=None):
    if type(bins) is np.ndarray:
        increment = bins[1] - bins[0]
        bin_edges = bins - 0.5 * increment
        upper_edge = bin_edges[-1] + increment
        bin_edges = np.append(bin_edges, upper_edge)
    return np.histogram(a, bins=bin_edges, range=range, normed=normed, weights=weights, density=density)[0]

    #bin_edges is undefined if bins isn't numpy array. Can bins not be an array?