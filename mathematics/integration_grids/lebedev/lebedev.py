'''
The grid points were taken from the sphericalquadpy project (Thomas Camminady):
https://github.com/camminady/sphericalquadpy
'''

import numpy as np

# A dictionary of available degrees of spherical harmonics and 
# corresponding numbers of grid points
lebedev_num_points_by_degree = {
    3 : 6, 
    5 : 14, 
    7 : 26, 
    9 : 38, 
    11 : 50, 
    13 : 74, 
    15 : 86, 
    17 : 110,
    19 : 146, 
    21 : 170, 
    23 : 194, 
    25 : 230, 
    27 : 266, 
    29 : 302, 
    31 : 350, 
    35 : 434,
    41 : 590, 
    47 : 770, 
    53 : 974, 
    59 : 1202, 
    65 : 1454, 
    71 : 1730, 
    77 : 2030, 
    83 : 2354,
    89 : 2702, 
    95 : 3074, 
    101 : 3470, 
    107 : 3890, 
    113 : 4334, 
    119 : 4802, 
    125 : 5294, 
    131 : 5810
    }   

# A dictionary of the grid points, degree:[x,y,z,weight]
def lebedev_grid_points():
    grid_points = {}
    for key in lebedev_num_points_by_degree:
        filename = "mathematics/integration_grids/lebedev/%s_lebedev.txt" % str(key)
        xyzw = np.loadtxt(filename, delimiter=",") 
        grid_points[key] = xyzw
    return grid_points


# grid_points = lebedev_grid_points()
# print(grid_points)