import numpy as np
from scipy import integrate
import os
import sys
sys.path.append(os.getcwd())
from mathematics.integration_grids.gauss_legendre_quadrature import GaussLegendreQuadrature
from mathematics.integration_grids.tests.error_vs_num_points import error_vs_num_points, plot_error_vs_num_points

# distributions
def uniform_distribution(x, mean, width):
    if width == 0:
        return np.where((x >= mean-0.5*width) & (x <= mean+0.5*width), 1.0, 0.0)
    else:
        return np.where((x >= mean-0.5*width) & (x <= mean+0.5*width), 1/width, 0.0)

def normal_distribution(x, mean, width):
    if width == 0:
        return np.where(x == mean, 1.0, 0.0)
    else:
        return np.exp(-0.5 * ((x - mean)/width)**2) / (np.sqrt(2*np.pi) * width)


# test integrands
def test_integrand0(x):
    return uniform_distribution(x,5.0,2.0)

def test_integrand1(x):
    return normal_distribution(x,3.0,0.05)

def test_integrand2(x):
    return 0.5 * normal_distribution(x,3.0,0.05) + 0.5 * normal_distribution(x,5.0,0.01)


# Set integration ranges
thereshold = 1e-6
x = np.linspace(1,8,80001)

y0 = test_integrand0(x)
idx_min, idx_max = min(np.where(y0 > thereshold)[0]), max(np.where(y0 > thereshold)[0])
x_min0, x_max0 = x[idx_min], x[idx_max] 

y1 = test_integrand1(x)
idx_min, idx_max = min(np.where(y1 > thereshold)[0]), max(np.where(y1 > thereshold)[0])
x_min1, x_max1 = x[idx_min], x[idx_max] 

y2 = test_integrand2(x)
idx_min, idx_max = min(np.where(y2 > thereshold)[0]), max(np.where(y2 > thereshold)[0])
x_min2, x_max2 = x[idx_min], x[idx_max] 


# Compute integrals with scipy 
TI0 = integrate.quad(test_integrand0, x_min0, x_max0)[0]
TI1 = integrate.quad(test_integrand1, x_min1, x_max1)[0]
TI2 = integrate.quad(test_integrand2, x_min2, x_max2)[0]


# Compute integrals with the Gauss-Legendre quadrature
grid = GaussLegendreQuadrature()
number_of_points = 100
TI0_grid = grid.integrate_function(test_integrand0, number_of_points, x_min0, x_max0)
TI1_grid = grid.integrate_function(test_integrand1, number_of_points, x_min1, x_max1)
TI2_grid = grid.integrate_function(test_integrand2, number_of_points, x_min2, x_max2)


# Print results
print("Test functions:")
print("Test Integrand 0: f(x) = Uniform(5,2)")
print("Test Integrand 1: f(x) = Gaussian(3,0.05)")
print("Test Integrand 2: f(x) = 0.5*Gaussian(3,0.05)+0.5*Gaussian(5,0.1)")

print("\nScipy quad:")
print("Test Integrand 0: " + str(TI0))
print("Test Integrand 1: " + str(TI1))
print("Test Integrand 2: " + str(TI2))

print(f"\nGauss-Legendre quadrature with {number_of_points} points: ")
print("Test Integrand 0: " + str(TI0_grid)+ " - Deviation: " + str(abs(100*(TI0-TI0_grid)/TI0)) + "%")
print("Test Integrand 1: " + str(TI1_grid)+ " - Deviation: " + str(abs(100*(TI1-TI1_grid)/TI1)) + "%")
print("Test Integrand 2: " + str(TI2_grid)+ " - Deviation: " + str(abs(100*(TI2-TI2_grid)/TI2)) + "%")


# Compute the error of the grid-based integration vs number of grid points
print("\nComputing the integration error vs number of grid points...")
num_points_array = np.arange(1, 201, 10)
errors0 = error_vs_num_points(grid, test_integrand0, TI0, num_points_array, [x_min0, x_max0])
errors1 = error_vs_num_points(grid, test_integrand1, TI1, num_points_array, [x_min1, x_max1])
errors2 = error_vs_num_points(grid, test_integrand2, TI2, num_points_array, [x_min2, x_max2])
data = [errors0, errors1, errors2] 
labels = ["f(x) = Uniform(5,2)", "f(x) = Gaussian(3,0.05)", "0.5*Gaussian(3,0.05)+0.5*Gaussian(5,0.1)"]
titel = "Integration using Gauss-Legendre grids"
titel += "\n\nReference integral values were obtained using scipy.quad"
filename = 'errors_gauss_legendre'
plot_error_vs_num_points(data, labels, titel, filename)