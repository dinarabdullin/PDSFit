'''
Fibonacci grid
'''

import numpy as np

def fibonacci_grid_points(size):
	v = []
	phi = np.pi * (3. - np.sqrt(5.))        # golden angle in radians
	for i in range(size):
		y = 1 - (i / float(size - 1)) * 2   # y goes from 1 to -1
		radius = np.sqrt(1 - y * y)         # radius at y
		theta = phi * i                     # golden angle increment
		x = np.cos(theta) * radius
		z = np.sin(theta) * radius
		v.append([x, y, z])
	return np.array(v)