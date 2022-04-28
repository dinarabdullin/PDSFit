''' Generate random points from a defined distribution '''

import numpy as np
from scipy import interpolate
from scipy.special import i0
try:
    from mathematics.distributions import sine_weighted_uniform_distribution, sine_weighted_normal_distribution, sine_weighted_vonmises_distribution
    from mathematics.histogram import histogram
except:
    from distributions import sine_weighted_uniform_distribution, sine_weighted_normal_distribution, sine_weighted_vonmises_distribution
    from histogram import histogram
    

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
        if width[0] == 0:
            points = np.repeat(mean[0], size)
        else:
            if not sine_weighted:
                points = np.random.uniform(mean[0]-0.5*width[0], mean[0]+0.5*width[0], size=size)
            else:
                args = {'mean': mean[0], 'width': width[0], 'ranges': np.array([0.0, np.pi]), 'samples': 100000}
                uniform_samples = np.random.random(size)
                points = random_points_from_arbitrary_distribution(sine_weighted_uniform_distribution, args)(uniform_samples)
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
                if width[i] == 0:
                    points_one_component = np.repeat(mean[i], size_components[i])
                else:
                    if not sine_weighted:
                        points_one_component = np.random.uniform(mean[i]-0.5*width[i], mean[i]+0.5*width[i], size_components[i])
                    else:
                        args = {'mean': mean[i], 'width': width[i], 'ranges': np.array([0.0, np.pi]), 'samples': 100000}
                        uniform_samples = np.random.random(size_components[i])
                        points_one_component = random_points_from_arbitrary_distribution(sine_weighted_uniform_distribution, args)(uniform_samples)    
                if first_component:
                    points = points_one_component
                    first_component = False
                else:
                    points = np.concatenate((points, points_one_component), axis=None)
    return points


def random_points_from_multimodal_normal_distribution(mean, width, rel_prob, size, sine_weighted):
    num_components = len(mean)
    if num_components == 1:
        if width[0] == 0:
            points = np.repeat(mean[0], size)
        else:
            if not sine_weighted:
                points = np.random.normal(mean[0], width[0], size=size)
            else:
                args = {'mean': mean[0], 'width': width[0], 'ranges': np.array([0.0, np.pi]), 'samples': 100000}
                uniform_samples = np.random.random(size)
                points = random_points_from_arbitrary_distribution(sine_weighted_normal_distribution, args)(uniform_samples)
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
                if width[i] == 0:
                    points_one_component = np.repeat(mean[i], size_components[i])
                else:
                    if not sine_weighted:
                        points_one_component = np.random.normal(mean[i], width[i], size_components[i])
                    else:
                        args = {'mean': mean[i], 'width': width[i], 'ranges': np.array([0.0, np.pi]), 'samples': 100000}
                        uniform_samples = np.random.random(size_components[i])
                        points_one_component = random_points_from_arbitrary_distribution(sine_weighted_normal_distribution, args)(uniform_samples)    
                if first_component:
                    points = points_one_component
                    first_component = False
                else:
                    points = np.concatenate((points, points_one_component), axis=None)
    return points


def random_points_from_multimodal_vonmises_distribution(mean, width, rel_prob, size, sine_weighted=False):
    num_components = len(mean)
    if num_components == 1:
        if (width[0] == 0) or (np.isfinite(i0(1/width[0]**2))== False):
            points = np.repeat(mean[0], size)
        else:
            if not sine_weighted:
                points = np.random.vonmises(mean[0], 1/width[0]**2, size=size)
            else:
                args = {'mean': mean[0], 'width': width[0], 'ranges': np.array([0.0, np.pi]), 'samples': 100000}
                uniform_samples = np.random.random(size)
                points = random_points_from_arbitrary_distribution(sine_weighted_vonmises_distribution, args)(uniform_samples)
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
                if (width[i] == 0) or (np.isfinite(i0(1/width[i]**2))== False):
                    points_one_component = np.repeat(mean[i], size_components[i])
                else:
                    if not sine_weighted:
                        points_one_component = np.random.vonmises(mean[i], 1/width[i]**2, size_components[i])
                    else:
                        args = {'mean': mean[i], 'width': width[i], 'ranges': np.array([0.0, np.pi]), 'samples': 100000}
                        uniform_samples = np.random.random(size_components[i])
                        points_one_component = random_points_from_arbitrary_distribution(sine_weighted_vonmises_distribution, args)(uniform_samples)    
                if first_component:
                    points = points_one_component
                    first_component = False
                else:
                    points = np.concatenate((points, points_one_component), axis=None)
    return points


def random_points_from_uniform_distribution(mean, width, size, sine_weighted=False):   
    if width[0] == 0:
        return np.repeat(mean[0], size)
    else:
        if not sine_weighted:
            return np.random.uniform(mean[0]-0.5*width[0], mean[0]+0.5*width[0], size=size)
        else:
            args = {'mean': mean[0], 'width': width[0], 'ranges': np.array([0.0, np.pi]), 'samples': 100000}
            uniform_samples = np.random.random(size)
            return random_points_from_arbitrary_distribution(sine_weighted_uniform_distribution, args)(uniform_samples)


def random_points_from_normal_distribution(mean, width, size, sine_weighted=False):
    if width[0] == 0:
        return np.repeat(mean[0])
    else:
        if not sine_weighted:
            return np.random.normal(mean[0], width[0], size=size)
        else:
            args = {'mean': mean[0], 'width': width[0], 'ranges': np.array([0.0, np.pi]), 'samples': 100000}
            uniform_samples = np.random.random(size)
            return random_points_from_arbitrary_distribution(sine_weighted_normal_distribution, args)(uniform_samples)


def random_points_from_vonmises_distribution(mean, width, size, sine_weighted=False):
    if (width[0] == 0) or (np.isfinite(i0(1/width[0]**2))== False):
        return np.repeat(mean[0], size)
    else:
        if not sine_weighted:
            return np.random.vonmises(mean[0], 1/width[0]**2, size=size)
        else:
            args = {'mean': mean[0], 'width': width[0], 'ranges': np.array([0.0, np.pi]), 'samples': 100000}
            uniform_samples = np.random.random(size)
            return random_points_from_arbitrary_distribution(sine_weighted_vonmises_distribution, args)(uniform_samples)


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


def test(): 
    
    # Unimodal distributions
    mean = [20 * np.pi/180]
    width = [90 * np.pi/180]
    rel_prob = []
    size = 1000000
    x11 = random_points_from_distribution("uniform", mean, width, rel_prob, size, sine_weighted=False)
    x12 = random_points_from_distribution("normal", mean, width, rel_prob, size, sine_weighted=False)
    x13 = random_points_from_distribution("vonmises", mean, width, rel_prob, size, sine_weighted=False)
    h11 = histogram(x11, bins=np.arange(-2*np.pi,2*np.pi,np.pi/180), density=True)
    h12 = histogram(x12, bins=np.arange(-2*np.pi,2*np.pi,np.pi/180), density=True)
    h13 = histogram(x13, bins=np.arange(-2*np.pi,2*np.pi,np.pi/180), density=True)
    
    # Bimodal distributions
    mean = [30 * np.pi/180, 90 * np.pi/180] 
    width = [10 * np.pi/180, 20 * np.pi/180]
    rel_prob = [0.5]
    size = 1000000
    x21 = random_points_from_distribution("normal", mean, width, rel_prob, size, sine_weighted=False)
    x22 = random_points_from_distribution("vonmises", mean, width, rel_prob, size, sine_weighted=False)
    h21 = histogram(x21, bins=np.arange(-2*np.pi,2*np.pi,np.pi/180), density=True)
    h22 = histogram(x22, bins=np.arange(-2*np.pi,2*np.pi,np.pi/180), density=True)
    
    # Unimodal sine-weighted distributions
    mean = [90 * np.pi/180]
    width = [90 * np.pi/180]
    rel_prob = []
    size = 1000000
    x31 = random_points_from_distribution("uniform", mean, width, rel_prob, size, sine_weighted=True)
    x32 = random_points_from_distribution("normal", mean, width, rel_prob, size, sine_weighted=True)
    x33 = random_points_from_distribution("vonmises", mean, width, rel_prob, size, sine_weighted=True)
    h31 = histogram(x31, bins=np.arange(-2*np.pi,2*np.pi,np.pi/180), density=True)
    h32 = histogram(x32, bins=np.arange(-2*np.pi,2*np.pi,np.pi/180), density=True)
    h33 = histogram(x33, bins=np.arange(-2*np.pi,2*np.pi,np.pi/180), density=True)
    
    # Bimodal sine-weighted distributions
    mean = [30 * np.pi/180, 90 * np.pi/180] 
    width = [10 * np.pi/180, 20 * np.pi/180]
    rel_prob = [0.5]
    size = 1000000
    x41 = random_points_from_distribution("normal", mean, width, rel_prob, size, sine_weighted=True)
    x42 = random_points_from_distribution("vonmises", mean, width, rel_prob, size, sine_weighted=True)
    h41 = histogram(x41, bins=np.arange(-2*np.pi,2*np.pi,np.pi/180), density=True)
    h42 = histogram(x42, bins=np.arange(-2*np.pi,2*np.pi,np.pi/180), density=True)
    
    # Plot
    import matplotlib.pyplot as plt
    fig = plt.figure(facecolor='w', edgecolor='w')
    axes1 = fig.add_subplot(221)
    axes1.plot(np.arange(-2*np.pi,2*np.pi,np.pi/180),h11)
    axes1.plot(np.arange(-2*np.pi,2*np.pi,np.pi/180),h12)
    axes1.plot(np.arange(-2*np.pi,2*np.pi,np.pi/180),h13)
    axes1.set_xlabel('x')
    axes1.set_ylabel('Probability density')
    axes1.legend(('uniform', 'normal', 'vonmises'))
    axes1.title.set_text('Unimodal distributions')
    axes2 = fig.add_subplot(222)
    axes2.plot(np.arange(-2*np.pi,2*np.pi,np.pi/180),h21)
    axes2.plot(np.arange(-2*np.pi,2*np.pi,np.pi/180),h22)
    axes2.set_xlabel('x')
    axes2.set_ylabel('Probability density')
    axes2.legend(('normal', 'vonmises'))
    axes2.title.set_text('Bimodal distributions')
    axes3 = fig.add_subplot(223)
    axes3.plot(np.arange(-2*np.pi,2*np.pi,np.pi/180),h31)
    axes3.plot(np.arange(-2*np.pi,2*np.pi,np.pi/180),h32)
    axes3.plot(np.arange(-2*np.pi,2*np.pi,np.pi/180),h33)
    axes3.set_xlabel('x')
    axes3.set_ylabel('Probability density')
    axes3.legend(('uniform', 'normal', 'vonmises'))
    axes3.title.set_text('Unimodal sine-weighted distributions')
    axes4 = fig.add_subplot(224)
    axes4.plot(np.arange(-2*np.pi,2*np.pi,np.pi/180),h41)
    axes4.plot(np.arange(-2*np.pi,2*np.pi,np.pi/180),h42)
    axes4.set_xlabel('x')
    axes4.set_ylabel('Probability density')
    axes4.legend(('normal', 'vonmises'))
    axes4.title.set_text('Bimodal sine-weighted distributions')
    plt.show()

# To test the random number generators, uncommment the next line and run this python script
#test()