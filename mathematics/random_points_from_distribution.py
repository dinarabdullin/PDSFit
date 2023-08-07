import numpy as np
from scipy import interpolate
from scipy.special import i0
from mathematics.distributions import sine_weighted_uniform_distribution, sine_weighted_normal_distribution, sine_weighted_vonmises_distribution
from mathematics.histogram import histogram
from supplement.definitions import const
 

def random_points_from_distribution(distribution_type, mean, width, rel_prob, size, sine_weighted):
    if distribution_type == "uniform":
        return random_points_from_multimodal_uniform_distribution(mean, width, rel_prob, size, sine_weighted)
    elif distribution_type == "normal":
        return random_points_from_multimodal_normal_distribution(mean, width, rel_prob, size, sine_weighted)
    elif distribution_type == "vonmises":
        return random_points_from_multimodal_vonmises_distribution(mean, width, rel_prob, size, sine_weighted)


def random_points_from_multimodal_uniform_distribution(mean, width, rel_prob, size, sine_weighted):
    num_components = len(mean)
    if num_components == 1:
        points = random_points_from_uniform_distribution(mean[0], width[0], size, sine_weighted)
    else:
        size_components = []
        for i in range(num_components-1):
            size_component = int(size * rel_prob[i])
            size_components.append(size_component)
        size_component = size - sum(size_components)
        size_components.append(size_component)
        first_component = True
        for i in range(num_components):
            if size_components[i] > 0:
                points_one_component = random_points_from_uniform_distribution(mean[i], width[i], size_components[i], sine_weighted)   
                if first_component:
                    points = points_one_component
                    first_component = False
                else:
                    points = np.concatenate((points, points_one_component), axis=None)
    return points


def random_points_from_multimodal_normal_distribution(mean, width, rel_prob, size, sine_weighted):
    num_components = len(mean)
    if num_components == 1:
        points = random_points_from_normal_distribution(mean[0], width[0], size, sine_weighted)
    else:
        size_components = []
        for i in range(num_components-1):
            size_component = int(size * rel_prob[i])
            size_components.append(size_component)
        size_component = size - sum(size_components)
        size_components.append(size_component)
        first_component = True
        for i in range(num_components):
            if size_components[i] > 0:
                points_one_component = random_points_from_normal_distribution(mean[i], width[i], size_components[i], sine_weighted)  
                if first_component:
                    points = points_one_component
                    first_component = False
                else:
                    points = np.concatenate((points, points_one_component), axis=None)
    return points


def random_points_from_multimodal_vonmises_distribution(mean, width, rel_prob, size, sine_weighted=False):
    num_components = len(mean)
    if num_components == 1:
        points = random_points_from_vonmises_distribution(mean[0], width[0], size, sine_weighted)
    else:
        size_components = []
        for i in range(num_components-1):
            size_component = int(size * rel_prob[i])
            size_components.append(size_component)
        size_component = size - sum(size_components)
        size_components.append(size_component)
        first_component = True
        for i in range(num_components):
            if size_components[i] > 0:
                points_one_component = random_points_from_vonmises_distribution(mean[i], width[i], size_components[i], sine_weighted)    
                if first_component:
                    points = points_one_component
                    first_component = False
                else:
                    points = np.concatenate((points, points_one_component), axis=None)
    return points


def random_points_from_uniform_distribution(mean, width, size, sine_weighted=False):   
    if width < 1e-9:
        return np.repeat(mean, size)
    else:
        if not sine_weighted:
            return np.random.uniform(mean-0.5*width, mean+0.5*width, size=size)
        else:
            args = {'mean': mean, 'width': width, 'ranges': np.array([0, np.pi]), 'samples': 100000}
            uniform_samples = np.random.random(size)
            return random_points_from_arbitrary_distribution(sine_weighted_uniform_distribution, args)(uniform_samples)


def random_points_from_normal_distribution(mean, width, size, sine_weighted=False):
    std = width * const['fwhm2std']
    if std < 1e-9:
        return np.repeat(mean, size)
    else:
        if not sine_weighted:
            return np.random.normal(mean, std, size=size)
        else:
            args = {'mean': mean, 'std': std, 'ranges': np.array([0, np.pi]), 'samples': 100000}
            uniform_samples = np.random.random(size)
            return random_points_from_arbitrary_distribution(sine_weighted_normal_distribution, args)(uniform_samples)


def random_points_from_vonmises_distribution(mean, width, size, sine_weighted=False):
    std = width * const['fwhm2std']
    if std < 1e-9:
        return np.repeat(mean, size)
    else:
        kappa =  1 / std**2
        if np.isfinite(i0(kappa)):
            if not sine_weighted:
                return np.random.vonmises(mean, kappa, size=size)
            else:
                args = {'mean': mean, 'std': std, 'ranges': np.array([0, np.pi]), 'samples': 100000}
                uniform_samples = np.random.random(size)
                return random_points_from_arbitrary_distribution(sine_weighted_vonmises_distribution, args)(uniform_samples)
        else:
            if not sine_weighted:
                return np.random.normal(mean, std, size=size)
            else:
                args = {'mean': mean, 'std': std, 'ranges': np.array([0, np.pi]), 'samples': 100000}
                uniform_samples = np.random.random(size)
                return random_points_from_arbitrary_distribution(sine_weighted_normal_distribution, args)(uniform_samples)


def random_points_from_arbitrary_distribution(f, args):
    '''
    Implementation of Inverse Transform Sampling for an arbitrary probability distribution.
    Theory: https://www.av8n.com/physics/arbitrary-probability.htm
    Implementation is based on: https://gist.github.com/amarvutha/c2a3ea9d42d238551c694480019a6ce1
    '''
    x = np.linspace(args['ranges'][0], args['ranges'][1], args['samples'])
    # Probability density function, pdf
    y = f(x, args) 
    # Cumulative distribution function, cdf
    cdf_y = np.cumsum(y)            
    cdf_y = cdf_y / cdf_y.max()
    # Inverse of the cumulative distribution function
    inverse_cdf = interpolate.interp1d(cdf_y,x)
    return inverse_cdf