''' Distributions '''

import numpy as np
import scipy
from scipy.special import i0

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


def vonmises_distribution(v, mean, std):
    if std[0] == 0:
        return np.where(v == mean[0], 1.0, 0.0)
    else:
        kappa =  1 / std[0]**2
        return np.exp(kappa * np.cos(v - mean[0])) / (2*np.pi * i0(kappa))


def multimodal_vonmises_distribution(v, mean, std, rel_prob):
    if len(mean) == 1:
        if std[0] == 0:
            return np.where(v == mean[0], 1.0, 0.0)
        else:
            kappa =  1 / std[0]**2
            return np.exp(kappa * np.cos(v - mean[0])) / (2*np.pi * i0(kappa))    
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
                kappa =  1 / std[i]**2
                result = result + weight * np.exp(kappa * np.cos(v - mean[i])) / (2*np.pi * i0(kappa)) 
        return result