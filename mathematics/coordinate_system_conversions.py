import numpy as np


def spherical2cartesian(rho, pol, azi):
    """Convert spherical coordinates into Cartestian coordinates."""
    x = rho * np.sin(pol) * np.cos(azi)
    y = rho * np.sin(pol) * np.sin(azi)
    z = rho * np.cos(pol)
    v = np.column_stack((x, y, z))
    return v


def cartesian2spherical(v):
    """Convert Cartesian coordinates into spherical coordinates. 
    rho : arbitrary range
    pol : [0, pi]
    azi : [-pi, pi]
    """
    xy = v[:, 0]**2 + v[:, 1]**2
    rho = np.sqrt(xy + v[:, 2]**2)
    pol = np.arctan2(np.sqrt(xy), v[:,2])
    azi = np.arctan2(v[:,1], v[:,0])
    azi = np.where(pol >= 0, azi, azi + np.pi)
    azi = np.where(azi <= np.pi, azi, azi - 2 * np.pi)
    pol = np.where(pol >= 0, pol, -1 * pol)
    return rho, pol, azi