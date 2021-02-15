'''
Convert spherical coordinates into Cartestian coordinates
'''

import numpy as np


# this took more than 15s
def spherical2cartesian(r, theta, phi):
	N = r.size
	v = np.zeros((N,3))
	for i in range(N):
		v[i][0] = r[i] * np.sin(theta[i]) * np.cos(phi[i])
		v[i][1] = r[i] * np.sin(theta[i]) * np.sin(phi[i])
		v[i][2] = r[i] * np.cos(theta[i])
	return v

#for loop could also be replaced with numpy

def spherical2cartesian_alternative(r, theta, phi):
	x = r * np.sin(theta) * np.cos(phi)
	y = r * np.sin(theta) * np.sin(phi)
	z = r * np.cos(theta)
	v = np.column_stack((x, y, z))
	return v
