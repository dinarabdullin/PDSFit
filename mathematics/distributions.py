import numpy as np
import scipy
from scipy.special import i0


def uniform_distribution(x, args):
    """A uniform distribution"""
    mean = args["mean"]
    width = args["width"]
    if width == 0:
        return np.where(x == mean, 1, 0)
    else:
        return np.where((x >= mean - 0.5 * width) & (x <= mean + 0.5 * width), 1 / width, 0)


def normal_distribution(x, args):
    """A Gaussian distribution"""
    mean = args["mean"]
    std = args["std"]
    if std == 0:
        return np.where(x == mean, 1, 0)
    else:
        return np.exp(-0.5 * ((x - mean) / std)**2) / (np.sqrt(2 * np.pi) * std)


def vonmises_distribution(x, args):
    """A von Mises distribution"""
    mean = args["mean"]
    std = args["std"]
    if std == 0:
        return np.where(x == mean, 1, 0)
    else:
        kappa =  1 / std**2
        if np.isfinite(i0(kappa)):
            return np.exp(kappa * np.cos(x - mean)) / (2 * np.pi * i0(kappa))
        else:
            return np.exp(-0.5 * ((x - mean) / std)**2) / (np.sqrt(2 * np.pi) * std)


def sine_weighted_uniform_distribution(x, args):
    """A sine-weighted uniform distribution"""
    mean = args["mean"]
    width = args["width"]
    if width == 0:
        return np.where(x == mean, 1, 0)
    else:
        return np.where((x >= mean - 0.5 * width) & (x <= mean + 0.5 * width), 1 / width, 0) * np.abs(np.sin(x))


def sine_weighted_normal_distribution(x, args):
    """A sine-weighted Gaussian distribution"""
    mean = args["mean"]
    std = args["std"]
    if std == 0:
        return np.where(x == mean, 1, 0)
    else:
        return np.exp(-0.5 * ((x - mean) / std)**2) / (np.sqrt(2 * np.pi) * std) * np.abs(np.sin(x))    


def sine_weighted_vonmises_distribution(x, args):
    """A sine-weighted von Mises distribution"""
    mean = args["mean"]
    std = args["std"]
    if std == 0:
        return np.where(x == mean, 1, 0)
    else:
        kappa =  1 / std**2
        if np.isfinite(i0(kappa)):
            return np.exp(kappa * np.cos(x - mean)) / (2 * np.pi * i0(kappa)) * np.abs(np.sin(x))
        else:
            return np.exp(-0.5 * ((x - mean) / std)**2) / (np.sqrt(2 * np.pi) * std) * np.abs(np.sin(x))    