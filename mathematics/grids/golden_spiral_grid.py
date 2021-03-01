''' Golden spiral grid '''

import numpy as np

def golden_spiral_grid_points(size):
    v = []
    indices = np.arange(0, size, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/float(size))
    theta = np.pi * (1 + 5**0.5) * indices
    for i in range(size):
        x, y, z = np.cos(theta[i]) * np.sin(phi[i]), np.sin(theta[i]) * np.sin(phi[i]), np.cos(phi[i])
        v.append([x, y, z])
    v = np.array(v)
    w = (1/float(size)) * np.ones(size)
    return v, w