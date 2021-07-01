import os
import sys
import numpy as np

# A dictionary of available resolutions and
# the corresponding numbers of grid points:
# num_points = 12 * 6 * 2**(3*resolution)
mitchell_num_points_by_resolution = {
    0: 72,
    1: 576, 
    2: 4608, 
    3: 36864, 
    4: 294912, 
    5: 2359296
    }

# A dictionary of the grid points provided by quaternions, resolution:[a,b,c,d]
def mitchell_grid_points():
    grid_points = {}
    for key in mitchell_num_points_by_resolution:
        filename = "mathematics/integration_grids/mitchell/simple_grid_%s.qua" % str(key)
        quat = np.loadtxt(filename) 
        grid_points[key] = quat
    return grid_points


# grid_points = mitchell_grid_points()
# print(grid_points)