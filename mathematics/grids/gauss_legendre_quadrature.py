''' Gauss-Legendre quadrature '''

import numpy as np


def gauss_legendre_quadrature(lower_bound, upper_bound, deg):
    v, w = np.polynomial.legendre.leggauss(deg)
    v = 0.5*(v + 1)*(upper_bound - lower_bound) + lower_bound
    w *= 0.5*(upper_bound - lower_bound)
    return v, w