''' Distributions '''

import numpy as np
import scipy
from scipy.special import i0


def uniform_distribution(x, args):
    mean = args['mean'][0]
    width = args['width'][0]
    if width == 0:
        return np.where((x >= mean-0.5*width) & (x <= mean+0.5*width), 1.0, 0.0)
    else:
        return np.where((x >= mean-0.5*width) & (x <= mean+0.5*width), 1/width, 0.0)


def normal_distribution(x, args):
    mean = args['mean'][0]
    width = args['width'][0]
    if width == 0:
        return np.where(x == mean, 1.0, 0.0)
    else:
        return np.exp(-0.5 * ((x - mean)/width)**2) / (np.sqrt(2*np.pi) * width)


def vonmises_distribution(x, args):
    mean = args['mean'][0]
    width = args['width'][0]
    if width == 0:
        return np.where(x == mean, 1.0, 0.0)
    else:
        kappa =  1 / width**2
        if np.isfinite(i0(kappa)):
            return np.exp(kappa * np.cos(x - mean)) / (2*np.pi * i0(kappa))
        else:
            return np.where(x == mean, 1.0, 0.0)


def multimodal_normal_distribution(x, args):
    mean = args['mean']
    width = args['width']
    rel_prob = args['rel_prob']
    num_components = len(mean)
    if num_components == 1:
        if width[0] == 0:
            return np.where(x == mean[0], 1.0, 0.0)
        else:
            return np.exp(-0.5 * ((x - mean[0])/width[0])**2) / (np.sqrt(2*np.pi) * width[0])    
    else:
        last_weight = 1.0
        y = np.zeros(x.size)
        for i in range(num_components):
            if i < num_components - 1:
                weight = rel_prob[i]
                last_weight -= weight
            elif i == num_components - 1:
                weight = last_weight
            if width[i] == 0:
                y = y + weight * np.where(x == mean[i], 1.0, 0.0)
            else:   
                y = y + weight * np.exp(-0.5 * ((x - mean[i])/width[i])**2) / (np.sqrt(2*np.pi) * width[i])
        return y


def multimodal_vonmises_distribution(x, args):
    mean = args['mean']
    width = args['width']
    rel_prob = args['rel_prob']
    num_components = len(mean)
    if num_components == 1:
        if width[0] == 0:
            return np.where(x == mean[0], 1.0, 0.0)
        else:
            kappa =  1 / width[0]**2
            if np.isfinite(i0(kappa)):
                return np.exp(kappa * np.cos(x - mean[0])) / (2*np.pi * i0(kappa))
            else:
                return np.where(x == mean[0], 1.0, 0.0)
    else:
        last_weight = 1.0
        y = np.zeros(x.size)
        for i in range(num_components):
            if i < num_components - 1:
                weight = rel_prob[i]
                last_weight -= weight
            elif i == num_components - 1:
                weight = last_weight
            if width[i] == 0:
                y = y + weight * np.where(x == mean[i], 1.0, 0.0)
            else:   
                kappa =  1 / width[i]**2
                if np.isfinite(i0(kappa)):
                    y = y + weight * np.exp(kappa * np.cos(x - mean[i])) / (2*np.pi * i0(kappa)) 
                else:
                    y = y + weight * np.where(x == mean[i], 1.0, 0.0)
        return y


def sine_weigthed_uniform_distribution(x, args):
    mean = args['mean'][0]
    width = args['width'][0]
    if width == 0:
        return np.where((x >= mean-0.5*width) & (x <= mean+0.5*width), 1.0, 0.0) * np.abs(np.sin(x))
    else:
        return np.where((x >= mean-0.5*width) & (x <= mean+0.5*width), 1/width, 0.0) * np.abs(np.sin(x))


def sine_weighted_multimodal_normal_distribution(x, args):
    mean = args['mean']
    width = args['width']
    rel_prob = args['rel_prob']
    num_components = len(mean)
    if num_components == 1:
        if width[0] == 0:
            return np.where(x == mean[0], 1.0, 0.0)
        else:
            return np.exp(-0.5 * ((x - mean[0])/width[0])**2) / (np.sqrt(2*np.pi) * width[0]) * np.abs(np.sin(x))    
    else:
        last_weight = 1.0
        y = np.zeros(x.size)
        for i in range(num_components):
            if i < num_components - 1:
                weight = rel_prob[i]
                last_weight -= weight
            elif i == num_components - 1:
                weight = last_weight
            if width[i] == 0:
                y = y + weight * np.where(x == mean[i], 1.0, 0.0) * np.abs(np.sin(x))
            else:   
                y = y + weight * np.exp(-0.5 * ((x - mean[i])/width[i])**2) / (np.sqrt(2*np.pi) * width[i]) * np.abs(np.sin(x))
        return y


def sine_weighted_multimodal_vonmises_distribution(x, args):
    mean = args['mean']
    width = args['width']
    rel_prob = args['rel_prob']
    num_components = len(mean)
    if num_components == 1:
        if width[0] == 0:
            return np.where(x == mean[0], 1.0, 0.0)
        else:
            kappa =  1 / width[0]**2
            if np.isfinite(i0(kappa)):
                return np.exp(kappa * np.cos(x - mean[0])) / (2*np.pi * i0(kappa)) * np.abs(np.sin(x))
            else:
                return np.where(x == mean[0], 1.0, 0.0)
    else:
        last_weight = 1.0
        y = np.zeros(x.size)
        for i in range(num_components):
            if i < num_components - 1:
                weight = rel_prob[i]
                last_weight -= weight
            elif i == num_components - 1:
                weight = last_weight
            if width[i] == 0:
                y = y + weight * np.where(x == mean[i], 1.0, 0.0) * np.abs(np.sin(x))
            else:   
                kappa =  1 / width[i]**2
                if np.isfinite(i0(kappa)):
                    y = y + weight * np.exp(kappa * np.cos(x - mean[i])) / (2*np.pi * i0(kappa)) * np.abs(np.sin(x))
                else:
                    y = y + weight * np.where(x == mean[i], 1.0, 0.0) * np.abs(np.sin(x))
        return y
