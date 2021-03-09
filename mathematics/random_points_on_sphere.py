import numpy as np
from mathematics.coordinate_system_conversions import spherical2cartesian


def random_points_on_sphere(size):
    ''' Generate random points on a sphere '''
    r = np.ones(size)
    theta = np.arccos(2.0 * np.random.random_sample(size) - 1.0)
    phi = 2.0 * np.pi * np.random.random_sample(size)
    v = spherical2cartesian(r, theta, phi)
    return v