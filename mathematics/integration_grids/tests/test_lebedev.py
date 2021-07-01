import os
import sys
import numpy as np
import scipy
from scipy.special import i0
from scipy import integrate
sys.path.append(os.getcwd())
from mathematics.integration_grids.lebedev_angular_quadrature import LebedevAngularQuadrature
from mathematics.integration_grids.tests.error_vs_num_points import error_vs_num_points, plot_error_vs_num_points


# distributions
deg2rad = np.pi / 180.0

def vonmises_distribution(x, mean, width):
    if width == 0:
        return np.where(x == mean, 1.0, 0.0)
    else:
        kappa =  1 / width**2
        if np.isfinite(i0(kappa)):
            return np.exp(kappa * np.cos(x - mean)) / (2*np.pi * i0(kappa))
        else:
            return np.where(x == mean, 1.0, 0.0)

def uniform_distribution(x, mean, width):
    if width == 0:
        return np.where((x >= mean-0.5*width) & (x <= mean+0.5*width), 1.0, 0.0)
    else:
        return np.where((x >= mean-0.5*width) & (x <= mean+0.5*width), 1/width, 0.0)


# test integrands
def test_integrand0(xi, phi, weighted = False):
    if type(xi) is np.ndarray: 
        f = np.ones(xi.shape)
    else:
        f = 1
    return f if not weighted else f * np.sin(xi)

def test_integrand1(xi, phi, weighted = False):
    f = uniform_distribution(xi,deg2rad*80,deg2rad*10)
    return f if not weighted else f * np.sin(xi)

def test_integrand2(xi, phi, weighted = False):
    f = uniform_distribution(phi,deg2rad*45,deg2rad*20) * uniform_distribution(xi,deg2rad*80,deg2rad*10)
    return f if not weighted else f * np.sin(xi)

def test_integrand3(xi, phi, weighted = False):
    f = vonmises_distribution(xi,deg2rad*80,deg2rad*10)
    return f if not weighted else f * np.sin(xi)

def test_integrand4(xi, phi, weighted = False):
    f = vonmises_distribution(phi,deg2rad*45,deg2rad*20) * vonmises_distribution(xi,deg2rad*80,deg2rad*10)
    return f if not weighted else f * np.sin(xi)


# Compute integrals with scipy and the weight factor
xi_range = [0, np.pi] 
phi_range = [0, 2*np.pi]
TI0 = integrate.dblquad(test_integrand0, phi_range[0], phi_range[1], lambda phi: xi_range[0], lambda phi: xi_range[1], args=(True,))[0]
TI1 = integrate.dblquad(test_integrand1, phi_range[0], phi_range[1], lambda phi: xi_range[0], lambda phi: xi_range[1], args=(True,))[0]
TI2 = integrate.dblquad(test_integrand2, phi_range[0], phi_range[1], lambda phi: xi_range[0], lambda phi: xi_range[1], args=(True,))[0]
TI3 = integrate.dblquad(test_integrand3, phi_range[0], phi_range[1], lambda phi: xi_range[0], lambda phi: xi_range[1], args=(True,))[0]
TI4 = integrate.dblquad(test_integrand4, phi_range[0], phi_range[1], lambda phi: xi_range[0], lambda phi: xi_range[1], args=(True,))[0]


# Compute integrals with Lebedev quadrature
grid = LebedevAngularQuadrature()
num_points = 100
TI0_grid = grid.integrate_function(test_integrand0, num_points)
TI1_grid = grid.integrate_function(test_integrand1, num_points)
TI2_grid = grid.integrate_function(test_integrand2, num_points)
TI3_grid = grid.integrate_function(test_integrand3, num_points)
TI4_grid = grid.integrate_function(test_integrand4, num_points)


# Display results
print("Test functions:")
print("Test integrand 0 : f(ξ,φ) = sin(ξ) ")
print("Test integrand 1 : f(ξ,φ) = Uniform(ξ,80,10) * sin(ξ)")
print("Test integrand 2 : f(ξ,φ) = Uniform(φ,45,20) * Uniform(ξ,80,10) * sin(ξ)")
print("Test integrand 3 : f(ξ,φ) = vonMises(ξ,80,10) * sin(ξ)")
print("Test integrand 4 : f(ξ,φ) = vonMises(φ,45,20) * vonMises(ξ,80,10) * sin(ξ)")
print()

print("Scipy dblquad: ")
print("Test integrand 0 : " + str(TI0))
print("Test integrand 1 : " + str(TI1))
print("Test integrand 2 : " + str(TI2))
print("Test integrand 3 : " + str(TI3))
print("Test integrand 4 : " + str(TI4))
print()

print(f"Lebedev quadrature with {num_points} points: ")
print("Test Integrand 0: " + str(TI0_grid) + " - Deviation: " + str(abs(100*(TI0-TI0_grid)/TI0)) + "%")
print("Test Integrand 1: " + str(TI1_grid) + " - Deviation: " + str(abs(100*(TI1-TI1_grid)/TI1)) + "%")
print("Test Integrand 2: " + str(TI2_grid) + " - Deviation: " + str(abs(100*(TI2-TI2_grid)/TI2)) + "%")
print("Test Integrand 3: " + str(TI3_grid) + " - Deviation: " + str(abs(100*(TI3-TI3_grid)/TI3)) + "%")
print("Test Integrand 4: " + str(TI4_grid) + " - Deviation: " + str(abs(100*(TI4-TI4_grid)/TI4)) + "%")


# Compute the error of the grid-based integration vs number of grid points
print("\nComputing the integration error vs number of grid points...")
num_points_array = list(grid.num_points_by_degree.values())
num_points_array = sorted(num_points_array)
errors0 = error_vs_num_points(grid, test_integrand0, TI0, num_points_array, [])
errors1 = error_vs_num_points(grid, test_integrand1, TI1, num_points_array, [])
errors2 = error_vs_num_points(grid, test_integrand2, TI2, num_points_array, [])
errors3 = error_vs_num_points(grid, test_integrand3, TI3, num_points_array, [])
errors4 = error_vs_num_points(grid, test_integrand4, TI4, num_points_array, [])
data = [errors0, errors1, errors2, errors3, errors4] 
labels = ["f(ξ,φ) = sin(ξ)", 
          "f(ξ,φ) = Uniform(ξ,80,10) * sin(ξ)", 
          "f(ξ,φ) = Uniform(φ,45,20) * Uniform(ξ,80,10) * sin(ξ)",
          "f(ξ,φ) = vonMises(ξ,80,10) * sin(ξ)",
          "f(ξ,φ) = vonMises(φ,45,20) * vonMises(ξ,80,10) * sin(ξ)"
          ]
titel = "Integration using Lebedev angular grids"
titel += "\n\nReference integral values were obtained using scipy.dblquad"
filename = 'errors_lebedev'
plot_error_vs_num_points(data, labels, titel, filename)