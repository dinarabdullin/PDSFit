''' Generate random points from a given distribution '''

import numpy as np

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


def random_points_from_uniform_distribution(mean, width, size):   
    if width[0] == 0:
        return np.array([mean[0]])
    else:
        return np.random.uniform(mean[0]-0.5*width[0], mean[0]+0.5*width[0], size=size)


def random_points_from_normal_distribution(mean, width, size):
    if width[0] == 0:
        return np.array([mean[0]])
    else:
        return np.random.normal(mean[0], width[0], size=size)


def random_points_from_multimodal_normal_distribution(mean, width, rel_prob, size):
    if all(w == 0 for w in width):
        if len(mean) == 1:
            points = np.array([mean[0]])
        else:
            num_components = len(mean)
            size_last_component = size
            points = []
            for i in range(num_components):
                if i < num_components - 1:
                    size_component = int(size * rel_prob[i])
                    size_last_component -= size_component
                elif i == num_components - 1:
                    size_component = size_last_component
                points_component = np.array([mean[i]])
                points.append(points_component)   
            points = np.array(mean)
    else:
        if len(mean) == 1:
            points = np.random.normal(mean[0], width[0], size=size)
        else:
            num_components = len(mean)
            size_last_component = size
            points = []
            for i in range(num_components):
                if i < num_components - 1:
                    size_component = int(size * rel_prob[i])
                    size_last_component = size_last_component - size_component
                elif i == num_components - 1:
                    size_component = size_last_component
                points_component = np.random.normal(mean[i], width[i], size=size_component)
                points.append(points_component)
            points = np.array(points)
    return points
    

def random_points_from_vonmises_distribution(mean, width, size):
    if width[0] == 0:
        return np.array([mean[0]])
    else:
        return np.random.vonmises(mean[0], 1/width[0]**2, size=size)


def random_points_from_multimodal_vonmises_distribution(mean, width, rel_prob, size):
    if all(w == 0 for w in width):
        if len(mean) == 1:
            points = np.array([mean[0]])
        else:
            num_components = len(mean)
            size_last_component = size
            points = []
            for i in range(num_components):
                if i < num_components - 1:
                    size_component = int(size * rel_prob[i])
                    size_last_component -= size_component
                elif i == num_components - 1:
                    size_component = size_last_component
                points_component = np.array([mean[i]])
                points.append(points_component)   
            points = np.array(mean)
    else:
        if len(mean) == 1:
            points = np.random.vonmises(mean[0], 1/width[0]**2, size=size)
        else:
            num_components = len(mean)
            size_last_component = size
            points = []
            for i in range(num_components):
                if i < num_components - 1:
                    size_component = int(size * rel_prob[i])
                    size_last_component = size_last_component - size_component
                elif i == num_components - 1:
                    size_component = size_last_component
                points_component = np.random.vonmises(mean[i], 1/width[i]**2, size=size_component)
                points.append(points_component)
            points = np.array(points)
    return points    