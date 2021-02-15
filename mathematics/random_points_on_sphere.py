'''
Generate random points on a sphere
'''
from time import time
import numpy as np

from mathematics.spherical2cartesian import spherical2cartesian
from mathematics.spherical2cartesian import spherical2cartesian_alternative


def random_points_on_sphere(size):
	a = time()
	r = np.ones(size)
	theta = np.arccos(2.0 * np.random.random_sample(size) - 1.0)
	phi = 2.0 * np.pi * np.random.random_sample(size)
	b = time()
	v = spherical2cartesian_alternative(r, theta, phi)
	c=time()
	print("prior: " + str(b-a))
	print("sph2car: " + str(c-b) )
	return v