''' Distributions '''

import numpy as np

def uniform_distribution(v, mean, width):
    return np.where(v >= mean[0]-0.5*width[0] and v <= mean[0]+0.5*width[0], 1/width[0], 0.0)


def normal_distribution(v, mean, std):
    if std[0] == 0:
        return np.where(v == mean[0], 1.0, 0.0)
    else:
        return np.exp(-0.5 * ((v - mean[0])/std[0])**2) / (sqrt(2*np.pi) * std[0])


def multimodal_normal_distribution(v, mean, std, rel_prob):
    if len(mean) == 1:
        if std[0] == 0:
            return np.where(v == mean[0], 1.0, 0.0)
        else:
            return np.exp(-0.5 * ((v - mean[0])/std[0])**2) / (sqrt(2*np.pi) * std[0])    
    else:
        num_components = len(mean)
        last_weight = 1.0
        result = np.zeros(v.size)
        for i in range(num_components):
            if i < num_components - 1:
                weight = rel_prob[i]
                last_weight -= weight
            elif i == num_components - 1:
                weight = last_weight
            if std[i] == 0:
                result = result + weight * np.where(v == mean[i], 1.0, 0.0)
            else:   
                result = result + weight * exp(-0.5 * ((v - mean[i])/std[i])**2) / (sqrt(2*np.pi) * std[i])
        return result