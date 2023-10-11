import numpy as np
from scipy import interpolate
from scipy.special import i0
from mathematics.distributions import sine_weighted_uniform_distribution, \
    sine_weighted_normal_distribution, sine_weighted_vonmises_distribution
from mathematics.histogram import histogram
from supplement.definitions import const
 

def random_samples_from_distribution(distribution_type, mean, width, size, sine_weighted):
    """Generate random samples from a distribution."""
    if distribution_type == "uniform":
        return random_samples_from_uniform_distribution(mean, width, size, sine_weighted)
    elif distribution_type == "normal":
        return random_samples_from_normal_distribution(mean, width, size, sine_weighted)
    elif distribution_type == "vonmises":
        return random_samples_from_vonmises_distribution(mean, width, size, sine_weighted)


def random_samples_from_uniform_distribution(mean, width, size, sine_weighted = False): 
    """Generate random samples from a uniform distribution."""
    if width < 1e-9:
        return np.repeat(mean, size)
    else:
        if not sine_weighted:
            return np.random.uniform(mean - 0.5 * width, mean + 0.5 * width, size = size)
        else:
            args = {"mean": mean, "width": width, "ranges": np.array([0, np.pi]), "samples": 100000}
            distr = random_samples_from_arbitrary_distribution(sine_weighted_uniform_distribution, args)
            uniform_samples = np.random.random(size)
            return distr(uniform_samples)


def random_samples_from_normal_distribution(mean, width, size, sine_weighted = False):
    """Generate random samples from a Gaussian distribution."""
    std = width * const["fwhm2std"]
    if std < 1e-9:
        return np.repeat(mean, size)
    else:
        if not sine_weighted:
            return np.random.normal(mean, std, size=size)
        else:
            args = {"mean": mean, "std": std, "ranges": np.array([0, np.pi]), "samples": 100000}
            distr = random_samples_from_arbitrary_distribution(sine_weighted_normal_distribution, args)
            uniform_samples = np.random.random(size)
            return distr(uniform_samples)


def random_samples_from_vonmises_distribution(mean, width, size, sine_weighted = False):
    """Generate random samples from a von Mises distribution."""
    std = width * const["fwhm2std"]
    if std < 1e-9:
        return np.repeat(mean, size)
    else:
        kappa =  1 / std**2
        if np.isfinite(i0(kappa)):
            if not sine_weighted:
                return np.random.vonmises(mean, kappa, size = size)
            else:
                args = {"mean": mean, "std": std, "ranges": np.array([0, np.pi]), "samples": 100000}
                distr = random_samples_from_arbitrary_distribution(sine_weighted_vonmises_distribution, args)
                uniform_samples = np.random.random(size)
                return distr(uniform_samples)
        else:
            if not sine_weighted:
                return np.random.normal(mean, std, size=size)
            else:
                args = {"mean": mean, "std": std, "ranges": np.array([0, np.pi]), "samples": 100000}
                distr = random_samples_from_arbitrary_distribution(sine_weighted_normal_distribution, args)
                uniform_samples = np.random.random(size)
                return distr(uniform_samples)


def random_samples_from_arbitrary_distribution(f, args):
    """Inverse Transform Sampling for an arbitrary probability distribution.
    Theory: https://www.av8n.com/physics/arbitrary-probability.htm
    Implementation is based on: https://gist.github.com/amarvutha/c2a3ea9d42d238551c694480019a6ce1."""
    x = np.linspace(args["ranges"][0], args["ranges"][1], args["samples"])
    y = f(x, args)
    cdf_y = np.cumsum(y)            
    cdf_y = cdf_y / cdf_y.max()
    inverse_cdf = interpolate.interp1d(cdf_y,x)
    return inverse_cdf