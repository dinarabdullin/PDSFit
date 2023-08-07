import numpy as np
from mathematics.coordinate_system_conversions import spherical2cartesian


def random_points_on_sphere(size):
    ''' Generates random points on a sphere '''
    r = np.ones(size)
    xi = np.arccos(2 * np.random.random_sample(size) - 1)
    phi = 2 * np.pi * np.random.random_sample(size)
    v = spherical2cartesian(r, xi, phi)
    return v