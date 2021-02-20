'''
Convert spherical coordinates into Cartestian coordinates
'''

import numpy as np


def spherical2cartesian(r, theta, phi):
	x = r * np.sin(theta) * np.cos(phi)
	y = r * np.sin(theta) * np.sin(phi)
	z = r * np.cos(theta)
	v = np.column_stack((x, y, z))
	return v
