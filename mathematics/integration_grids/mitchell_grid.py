import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.append(os.getcwd())
from mathematics.integration_grids.integration_grid import IntegrationGrid
from mathematics.integration_grids.mitchell.mitchell import mitchell_grid_points, mitchell_num_points_by_resolution


class MitchellGrid(IntegrationGrid):
    
    '''
    Mitchell's SO(3) grid
    Publication: 
    A. Yershova, S. Jain, S.M. Lavalle, J.C. Mitchell, Int J Rob Res. 2010 June 1; 29(7): 801–812. doi:10.1177/0278364909352700
    '''
    
    def __init__(self):
        self.grid_points = mitchell_grid_points()
        self.num_points_by_resolution = mitchell_num_points_by_resolution

    def get_points(self, num_points, euler_angles_convention="ZXZ"):
        resolution = self.resolution_from_num_points(num_points)
        quat = self.grid_points[resolution]
        R = Rotation.from_quat(quat)
        points = R.as_euler(euler_angles_convention, degrees=False)
        return points
    
    def resolution_from_num_points(self, num_points):
        resolution, new_num_points = min(self.num_points_by_resolution.items(), key=lambda kv: abs(kv[1] - num_points))
        return resolution

    def get_weights(self, num_points):
        return (8*np.pi**2) / float(num_points)

    def get_weighted_summands(self, function, num_points, euler_angles_convention="ZXZ"):
        points = self.get_points(num_points, euler_angles_convention)
        weights = self.get_weights(points.shape[0])
        alpha, beta, gamma = points[:,0], points[:,1], points[:,2]
        return weights * function(alpha, beta, gamma)
    
    def integrate_function(self, function, num_points, euler_angles_convention="ZXZ"):
        return np.sum(self.get_weighted_summands(function, num_points, euler_angles_convention))