''' Coordinate system conversions '''
import numpy as np


def spherical2cartesian(rho, xi, phi):
    ''' Converts spherical coordinates into Cartestian coordinates '''
    x = rho * np.sin(xi) * np.cos(phi)
    y = rho * np.sin(xi) * np.sin(phi)
    z = rho * np.cos(xi)
    v = np.column_stack((x, y, z))
    return v


def cartesian2spherical(v):
    ''' Converts Cartesian coordinates into spherical coordinates '''
    xy = v[:,0]**2 + v[:,1]**2
    rho = np.sqrt(xy + v[:,2]**2)
    xi = np.arctan2(np.sqrt(xy), v[:,2])
    phi = np.arctan2(v[:,1], v[:,0])
    return rho, xi, phi