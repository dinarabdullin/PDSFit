import os
import sys
import numpy as np
from math import ceil
sys.path.append(os.getcwd())
from mathematics.integration_grids.integration_grid import IntegrationGrid
from mathematics.integration_grids.lebedev.lebedev import lebedev_grid_points, lebedev_num_points_by_degree
from mathematics.coordinate_system_conversions import cartesian2spherical


class LebedevAngularQuadrature(IntegrationGrid):
    
    '''
    Lebedev angular quadrature
    It is used to calculate integrals of functions on the surface of a unit sphere: 
    ∫sinθ dθ ∫f(r,θ,φ) dφ = ∑ wi f(r, θi, φi) 
    The integration ranges are: theta = [0, pi], phi = [0, 2*pi]
    '''

    def __init__(self):
        self.grid_points = lebedev_grid_points()
        self.num_points_by_degree = lebedev_num_points_by_degree

    def get_points(self, num_points):
        degree = self.degree_from_num_points(num_points)
        return self.grid_points[degree][:, 0:3]
    
    def get_points_spherical(self, num_points):
        xyz_points = self.get_points(num_points)
        rho, theta, phi = cartesian2spherical(xyz_points)
        return np.vstack((theta, phi))

    def get_weights(self, num_points):
        degree = self.degree_from_num_points(num_points)
        weights = self.grid_points[degree][:, 3]
        weights *= 4 * np.pi / sum(weights)
        return weights

    def degree_from_num_points(self, num_points):
        degree, new_num_points = min(self.num_points_by_degree.items(), key=lambda kv: abs(kv[1] - num_points))
        return degree

    def get_weighted_summands(self, function, num_points):
        xyz_points = self.get_points(num_points)
        weights = self.get_weights(num_points)
        rho, theta, phi = cartesian2spherical(xyz_points)
        return weights * function(theta, phi)

    def integrate_function(self, function, num_points):
        return np.sum(self.get_weighted_summands(function, num_points))    