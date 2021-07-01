import os
import sys
import numpy as np
sys.path.append(os.getcwd())
from mathematics.integration_grids.integration_grid import IntegrationGrid


class GaussLaguerreQuadrature(IntegrationGrid):
    """
    Gauss-Laguerre quadrature for the integration of a one dimensional function
    over the interval [0, inf]  with the weight function f(x) = exp(-x)
    """

    def get_points(self, num_points):
        return np.polynomial.laguerre.laggauss(num_points)[0]

    def get_weights(self, num_points):
        return np.polynomial.laguerre.laggauss(num_points)[1] * np.exp(self.get_points(num_points))

    def get_weighted_summands(self, function, num_points):
        return function(self.get_points(num_points)) * self.get_weights(num_points)
        
    def integrate_function(self, function, num_points):
        return np.dot(function(self.get_points(num_points)), self.get_weights(num_points))