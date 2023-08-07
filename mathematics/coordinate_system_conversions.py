import numpy as np


def spherical2cartesian(rho, xi, phi):
    ''' Converts spherical coordinates into Cartestian coordinates '''
    x = rho * np.sin(xi) * np.cos(phi)
    y = rho * np.sin(xi) * np.sin(phi)
    z = rho * np.cos(xi)
    v = np.column_stack((x, y, z))
    return v


def cartesian2spherical(v):
    ''' Converts Cartesian coordinates into spherical coordinates 
    rho : arbitryry range
    xi  : [0, pi]
    phi : [-pi, pi]
    '''
    xy = v[:,0]**2 + v[:,1]**2
    rho = np.sqrt(xy + v[:,2]**2)
    xi = np.arctan2(np.sqrt(xy), v[:,2])
    phi = np.arctan2(v[:,1], v[:,0])
    phi = np.where(xi >= 0, phi, phi + np.pi)
    phi = np.where(phi <= np.pi, phi, phi - 2*np.pi)
    xi = np.where(xi >= 0, xi, -1*xi)
    return rho, xi, phi