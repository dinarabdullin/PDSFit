import numpy as np


def histogram(a, bins = 10, range = None, density = None, weights = None):
    """A modified histogram with 'bins' being bin centers instead of bin edges."""
    if type(bins) is np.ndarray:
        increment = bins[1] - bins[0]
        bin_edges = bins - 0.5 * increment
        upper_edge = bin_edges[-1] + increment
        bin_edges = np.append(bin_edges, upper_edge)
        return np.histogram(a, bins = bin_edges, range = range, density = density, weights = weights)[0]
    else:
        return np.histogram(a, bins = bins, range = range, density = density, weights = weights) 