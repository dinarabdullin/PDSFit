import os
import sys
import numpy as np
sys.path.append(os.getcwd())
from mathematics.integration_grids.integration_grid import IntegrationGrid


class GaussLegendreQuadrature(IntegrationGrid):
    """ 
    Gauss-Legendre quadrature for a one-dimensional finite integral with variable integration bounds
    """

    def get_weights(self, num_points, lower_bound=-1, upper_bound=1):
        weights = np.polynomial.legendre.leggauss(num_points)[1]
        weights = weights * (upper_bound-lower_bound)/2
        return weights

    def get_points(self, num_points, lower_bound=-1, upper_bound=1):
        points = np.polynomial.legendre.leggauss(num_points)[0]
        #variable transformation for change of interval from [-1,1] to [lower, upper]
        points = (upper_bound-lower_bound)/2 * points + (upper_bound+lower_bound)/2
        return points

    def get_weighted_summands(self, function, num_points, lower_bound=-1, upper_bound=1):
        weights = self.get_weights(num_points, lower_bound, upper_bound)
        points = self.get_points(num_points, lower_bound, upper_bound)
        return weights * function(points)

    def integrate_function(self, function, num_points, lower_bound=-1, upper_bound=1):
        return np.sum(self.get_weighted_summands(function, num_points, lower_bound, upper_bound))