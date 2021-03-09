''' Generate random points from a defined distribution '''

import numpy as np
from scipy import interpolate
try:
    from mathematics.distributions import sine_weigthed_uniform_distribution, sine_weighted_multimodal_normal_distribution, sine_weighted_multimodal_vonmises_distribution
except:
    from distributions import sine_weigthed_uniform_distribution, sine_weighted_multimodal_normal_distribution, sine_weighted_multimodal_vonmises_distribution

def random_points_from_uniform_distribution(mean, width, size):   
    if width[0] == 0:
        return np.repeat(mean[0], size)
    else:
        return np.random.uniform(mean[0]-0.5*width[0], mean[0]+0.5*width[0], size=size)


def random_points_from_normal_distribution(mean, width, size):
    if width[0] == 0:
        return np.repeat(mean[0])
    else:
        return np.random.normal(mean[0], width[0], size=size)


def random_points_from_vonmises_distribution(mean, width, size):
    if width[0] == 0:
        return np.repeat(mean[0], size)
    else:
        return np.random.vonmises(mean[0], 1/width[0]**2, size=size)


def random_points_from_multimodal_normal_distribution(mean, width, rel_prob, size):
    num_components = len(mean)
    if num_components == 1:
        if width[0] == 0:
            points = np.repeat(mean[0], size)
        else:
            points = np.random.normal(mean[0], width[0], size=size)
    else:
        size_last_component = size
        for i in range(num_components):
            if i < num_components - 1:
                size_one_component = int(size * rel_prob[i])
                size_last_component = size_last_component - size_one_component
            elif i == num_components - 1:
                size_one_component = size_last_component
            if width[i] == 0:
                points_one_component = np.repeat(mean[i], size_one_component)
            else:
                points_one_component = np.random.normal(mean[i], width[i], size=size_one_component)
            if i == 0:
                points = points_one_component
            else:
                points = np.concatenate((points, points_one_component), axis=None)
    return points
    

def random_points_from_multimodal_vonmises_distribution(mean, width, rel_prob, size):
    num_components = len(mean)
    if num_components == 1:
        if width[0] == 0:
            points = np.repeat(mean[0], size)
        else:
            points = np.random.vonmises(mean[0], 1/width[0]**2, size=size)
    else:
        size_last_component = size
        for i in range(num_components):
            if i < num_components - 1:
                size_one_component = int(size * rel_prob[i])
                size_last_component = size_last_component - size_one_component
            elif i == num_components - 1:
                size_one_component = size_last_component
            if width[i] == 0:
                points_one_component = np.repeat(mean[i], size_one_component)
            else:
                points_one_component = np.random.vonmises(mean[i], 1/width[i]**2, size=size_one_component)
            if i == 0:
                points = points_one_component
            else:
                points = np.concatenate((points, points_one_component), axis=None)
    return points


def random_points_from_distribution(distribution_type, mean, width, rel_prob, size):
    if distribution_type == "uniform":
        return random_points_from_uniform_distribution(mean, width, size)
    elif distribution_type == "normal":
        return random_points_from_multimodal_normal_distribution(mean, width, rel_prob, size)
    elif distribution_type == "vonmises":
        return random_points_from_multimodal_vonmises_distribution(mean, width, rel_prob, size)
    else:
        raise ValueError('Invalid type of distribution!')
        sys.exit(1)


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


def random_points_from_sine_weighted_distribution(distribution_type, mean, width, rel_prob, size):
    if all(w == 0 for w in width):
        return random_points_from_distribution(distribution_type, mean, width, rel_prob, size)
    else:
        args={}
        args['mean'] = mean
        args['width'] = width
        args['rel_prob'] = rel_prob
        args['ranges'] = np.array([0.0, np.pi])
        args['samples'] = 100000
        uniform_samples = np.random.random(size)
        if distribution_type == "uniform":
            return random_points_from_arbitrary_distribution(sine_weigthed_uniform_distribution, args)(uniform_samples)
        elif distribution_type == "normal":
            return random_points_from_arbitrary_distribution(sine_weighted_multimodal_normal_distribution, args)(uniform_samples)
        elif distribution_type == "vonmises":
            return random_points_from_arbitrary_distribution(sine_weighted_multimodal_vonmises_distribution, args)(uniform_samples)
        else:
            raise ValueError('Invalid type of distribution!')
            sys.exit(1)


def test():
    # Unimodal distributions
    mean = [30 * np.pi/180]
    width = [90 * np.pi/180]
    rel_prob = []
    size = 1000000
    x11 = random_points_from_distribution("uniform", mean, width, rel_prob, size)
    x12 = random_points_from_distribution("normal", mean, width, rel_prob, size)
    x13 = random_points_from_distribution("vonmises", mean, width, rel_prob, size)
    
    # Bimodal distributions
    mean = [30 * np.pi/180, 90 * np.pi/180] 
    width = [10 * np.pi/180, 20 * np.pi/180]
    rel_prob = [0.5]
    size = 1000000
    x21 = random_points_from_distribution("normal", mean, width, rel_prob, size)
    x22 = random_points_from_distribution("vonmises", mean, width, rel_prob, size)
    
    # Unimodal sine-weighted distributions
    mean = [30 * np.pi/180]
    width = [90 * np.pi/180]
    rel_prob = []
    size = 1000000
    x31 = random_points_from_sine_weighted_distribution("uniform", mean, width, rel_prob, size)
    x32 = random_points_from_sine_weighted_distribution("normal", mean, width, rel_prob, size)
    x33 = random_points_from_sine_weighted_distribution("vonmises", mean, width, rel_prob, size)
    
    # Bimodal sine-weighted distributions
    mean = [30 * np.pi/180, 90 * np.pi/180] 
    width = [10 * np.pi/180, 20 * np.pi/180]
    rel_prob = [0.5]
    size = 1000000
    x41 = random_points_from_sine_weighted_distribution("normal", mean, width, rel_prob, size)
    x42 = random_points_from_sine_weighted_distribution("vonmises", mean, width, rel_prob, size)
    
    # Plot
    import matplotlib.pyplot as plt
    fig = plt.figure(facecolor='w', edgecolor='w')
    axes1 = fig.add_subplot(221)
    axes1.hist(x11,bins='auto',stacked=True,density=True,range=(-2*np.pi,2*np.pi))
    axes1.hist(x12,bins='auto',stacked=True,density=True,range=(-2*np.pi,2*np.pi))
    axes1.hist(x13,bins='auto',stacked=True,density=True,range=(-2*np.pi,2*np.pi))
    axes1.set_xlabel('x')
    axes1.set_ylabel('Probability density')
    axes1.legend(('uniform', 'normal', 'vonmises'))
    axes1.title.set_text('Unimodal distributions')
    axes2 = fig.add_subplot(222)
    axes2.hist(x21,bins='auto',stacked=True,density=True,range=(-2*np.pi,2*np.pi))
    axes2.hist(x22,bins='auto',stacked=True,density=True,range=(-2*np.pi,2*np.pi))
    axes2.set_xlabel('x')
    axes2.set_ylabel('Probability density')
    axes2.legend(('normal', 'vonmises'))
    axes2.title.set_text('Bimodal distributions')
    axes3 = fig.add_subplot(223)
    axes3.hist(x31,bins='auto',stacked=True,density=True,range=(-2*np.pi,2*np.pi))
    axes3.hist(x32,bins='auto',stacked=True,density=True,range=(-2*np.pi,2*np.pi))
    axes3.hist(x33,bins='auto',stacked=True,density=True,range=(-2*np.pi,2*np.pi))
    axes3.set_xlabel('x')
    axes3.set_ylabel('Probability density')
    axes3.legend(('uniform', 'normal', 'vonmises'))
    axes3.title.set_text('Unimodal sine-weighted distributions')
    axes4 = fig.add_subplot(224)
    axes4.hist(x41,bins='auto',stacked=True,density=True,range=(-2*np.pi,2*np.pi))
    axes4.hist(x42,bins='auto',stacked=True,density=True,range=(-2*np.pi,2*np.pi))
    axes4.set_xlabel('x')
    axes4.set_ylabel('Probability density')
    axes4.legend(('normal', 'vonmises'))
    axes4.title.set_text('Bimodal sine-weighted distributions')
    plt.show()

# To test the random number generators, uncommment the next line and run this python script
# test()