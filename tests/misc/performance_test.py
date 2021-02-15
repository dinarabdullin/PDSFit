"""
The method initialize_field() from PeldorCalculator.cpp was implemented in two different ways,
once using pure Python with a for loop and once using numpy without a for loop.

Result: as expected, numpy method is much faster, up to factor x25 for n = 1 million
"""

import time
import numpy as np
from math import sin, cos, pi, acos

n = 1000000

#using a normal Python loop and Python lists
#6.4 seconds for n = 1 million
def initialize_field1():
	fieldDirA = []
	for i in range(n):
		random = np.random.uniform(0,1)
		fphi = 2 * pi * random
		fxi = acos(random)
		single_dir = []
		single_dir.append(sin(fxi)* cos(fphi))
		single_dir.append(sin(fxi)*sin(fphi))
		single_dir.append(cos(fxi))
		fieldDirA.append(single_dir)
	return fieldDirA

#using numpy, without a for loop
#0.3 seconds for n = 1 million
def initialize_field2():
	random = np.random.uniform(0, 1, n)
	fphi = 2 * pi * random
	fxi_temp1 = np.arccos(random)
	fxi_temp2 = pi-fxi_temp1
	fxi = np.where(random<0.5, fxi_temp1, fxi_temp2)
	#elementwise multiplication, not matrix multiplication!
	x = np.sin(fxi) * np.cos(fphi)
	y = np.sin(fxi) * np.sin(fphi)
	z = np.cos(fxi)
	#stack to create the array [(x[0], y[0], z[0]), x[1], y[1], z[1]), .....]
	fieldDirA = np.column_stack((x, y, z))
	return fieldDirA

start = time.time()
initialize_field2()
end = time.time()
print(end-start)