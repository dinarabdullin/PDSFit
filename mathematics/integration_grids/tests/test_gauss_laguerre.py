import numpy as np
from scipy import integrate
import os
import sys
sys.path.append(os.getcwd())
from mathematics.integration_grids.gauss_laguerre_quadrature import GaussLaguerreQuadrature


# test functions
def test_integrand0(x):
    return np.exp(-x)*x**2

def test_integrand1(x):
    return np.exp(-x**2/2)

def test_integrand2(x):
    return np.cos(x) * np.sin(x) * np.exp(-x)

def test_integrand3(x):
    return 1/(1+x**2) 


# compute integrals with scipy 
TI0 = integrate.quad(test_integrand0, 0, np.inf)[0]
TI1 = integrate.quad(test_integrand1, 0, np.inf)[0]
TI2 = integrate.quad(test_integrand2, 0, np.inf)[0]
TI3 = integrate.quad(test_integrand3, 0, np.inf)[0]

# compute integrals with gauss laguerre quadrature
grid = GaussLaguerreQuadrature()
number_of_points = 20
TI0_grid = grid.integrate_function(test_integrand0, number_of_points)
TI1_grid = grid.integrate_function(test_integrand1, number_of_points)
TI2_grid = grid.integrate_function(test_integrand2, number_of_points)
TI3_grid = grid.integrate_function(test_integrand3, number_of_points)

print("Test functions, integration from 0 to infinity:")
print("Test Integrand 0: f(x) = x² * exp(-x)")
print("Test Integrand 1: f(x) = exp(-x**2)")
print("Test Integrand 2: f(x) = sin(x)*cos(x)*exp(-x)")
print("Test Integrand 3: f(x) = 1/(1+x²)")

print("\nScipy quad:")
print("Test Integrand 0: " + str(TI0))
print("Test Integrand 1: " + str(TI1))
print("Test Integrand 2: " + str(TI2))
print("Test Integrand 3: " + str(TI3))

print(f"\nGauss Laguerre with {number_of_points} points: ")
print("Test Integrand 0: " + str(TI0_grid)+ " - Deviation: " + str(abs(100*(TI0-TI0_grid)/TI0)) + "%")
print("Test Integrand 1: " + str(TI1_grid)+ " - Deviation: " + str(abs(100*(TI1-TI1_grid)/TI1)) + "%")
print("Test Integrand 2: " + str(TI2_grid)+ " - Deviation: " + str(abs(100*(TI2-TI2_grid)/TI2)) + "%")
print("Test Integrand 3: " + str(TI3_grid)+ " - Deviation: " + str(abs(100*(TI3-TI3_grid)/TI3)) + "%")
