import os
import io
import wx
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'
rcParams['axes.facecolor']= 'white'
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'Arial'
rcParams['lines.linewidth'] = 1
rcParams['xtick.major.size'] = 4
rcParams['xtick.major.width'] = 1
rcParams['ytick.major.size'] = 4
rcParams['ytick.major.width'] = 1
rcParams['font.size'] = 18
import sys 
sys.path.append('..')
from supplement.definitions import const
from mathematics.histogram import histogram


def get_filepath(message):
    app = wx.App(None) 
    dialog = wx.FileDialog(None, message, wildcard='*.*', style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
    if dialog.ShowModal() == wx.ID_OK:
        filepath = dialog.GetPath()
    else:
        filepath = ''
    return filepath
    
    
def load_model_parameters(filepath):
    loaded_parameters = [] 
    file = open(filepath, 'r')
    next(file)
    for line in file:
        first_column = list(chunk_string(line[0:19], 20))
        next_columns = list(chunk_string(line[20:-1], 15))
        data = []
        data.extend(first_column)
        data.extend(next_columns)
        loaded_parameter = {}
        name = data[0].strip()
        name_found = False
        for key in const['model_parameter_names_and_units']:    
            if name == const['model_parameter_names_and_units'][key]:
                loaded_parameter['name'] = key
                name_found = True
        if not name_found:
            raise ValueError('Error is encountered in the file with the optimized parameters of the model!')
            sys.exit(1)
        loaded_parameter['component'] = int(data[1])
        optimized = data[2].strip()
        if optimized == 'yes':
            loaded_parameter['optimized'] = 1
        elif optimized == 'no':
            loaded_parameter['optimized'] = 0
        else:
            print('Error is encountered in the file with the optimized parameters of the model!')
        loaded_parameter['value'] = float(data[3])
        minus_error = data[4].strip()
        plus_error = data[5].strip()
        if minus_error == 'nan' or plus_error == 'nan':
            minus_error_value = np.nan
            plus_error_value = np.nan
        else:
            minus_error_value = float(minus_error)
            plus_error_value = float(plus_error)
        loaded_parameter['errors'] = np.array([minus_error_value, plus_error_value])
        loaded_parameters.append(loaded_parameter)  
    model_parameters = {}
    model_parameter_errors = {}
    for key in const['model_parameter_names']:
        model_parameters[key] = []
        model_parameter_errors[key] = []
        for loaded_parameter in loaded_parameters:
            if loaded_parameter['name'] == key:
                model_parameters[key].append(loaded_parameter['value'])
                model_parameter_errors[key].append(loaded_parameter['errors'])
    return model_parameters, model_parameter_errors


def chunk_string(string, length):
    return (string[0+i:length+i] for i in range(0, len(string), length))


def compute_distance_distribution(model_parameters, model_parameter_errors, sample_size=10000, r_minimum=15, r_maximum=80, r_increment=0.01):
    distance_distribution = {}
    distance_distribution['r'] = np.arange(r_minimum, r_maximum+r_increment, r_increment)
    distance_distribution['p'] = multimodal_normal_distribution(distance_distribution['r'], model_parameters['r_mean'], model_parameters['r_width'], model_parameters['rel_prob'])
    distance_distribution['pl'] = copy.deepcopy(distance_distribution['p'])
    distance_distribution['pu'] = copy.deepcopy(distance_distribution['p'])
    distance_distribution['pl_rm'] = copy.deepcopy(distance_distribution['p'])
    distance_distribution['pu_rm'] = copy.deepcopy(distance_distribution['p'])
    distance_distribution['pl_rw'] = copy.deepcopy(distance_distribution['p'])
    distance_distribution['pu_rw'] = copy.deepcopy(distance_distribution['p'])
    r_mean_values = generate_random_points(model_parameters['r_mean'], model_parameter_errors['r_mean'], sample_size)
    r_width_values = generate_random_points(model_parameters['r_width'], model_parameter_errors['r_width'], sample_size)
    rel_prob_values = generate_random_points(model_parameters['rel_prob'], model_parameter_errors['rel_prob'], sample_size)
    for i in range(sample_size):
        print(i)
        test_distance_distribution = multimodal_normal_distribution(distance_distribution['r'], r_mean_values[i], r_width_values[i], rel_prob_values[i])
        distance_distribution['pl'] = np.amin(np.array([distance_distribution['pl'], test_distance_distribution]), axis=0)
        distance_distribution['pu'] = np.amax(np.array([distance_distribution['pu'], test_distance_distribution]), axis=0)
        test_distance_distribution1 = multimodal_normal_distribution(distance_distribution['r'], 
                                                                     r_mean_values[i], 
                                                                     model_parameters['r_width'], 
                                                                     model_parameters['rel_prob'])
        distance_distribution['pl_rm'] = np.amin(np.array([distance_distribution['pl_rm'], test_distance_distribution1]), axis=0)
        distance_distribution['pu_rm'] = np.amax(np.array([distance_distribution['pu_rm'], test_distance_distribution1]), axis=0)
        test_distance_distribution2 = multimodal_normal_distribution(distance_distribution['r'], 
                                                                     model_parameters['r_mean'], 
                                                                     r_width_values[i], 
                                                                     model_parameters['rel_prob'])
        distance_distribution['pl_rw'] = np.amin(np.array([distance_distribution['pl_rw'], test_distance_distribution2]), axis=0)
        distance_distribution['pu_rw'] = np.amax(np.array([distance_distribution['pu_rw'], test_distance_distribution2]), axis=0)
    return distance_distribution


def multimodal_normal_distribution(x, mean, width, rel_prob):
    num_components = len(mean)
    std = const['fwhm2std'] * np.array(width)
    if num_components == 1:
        if std[0] < 1e-3:
            idx = np.searchsorted(x[:-1], mean[0], side="left")
            return np.where(x == x[idx], 1, 0)
        else:
            return np.exp(-0.5 * ((x - mean[0])/std[0])**2)
    else:
        p = np.zeros(x.size)
        for i in range(num_components):
            if i < num_components - 1:
                weight = rel_prob[i]
            else:
                weight = 1.0 - np.sum(rel_prob)
            if std[i] < 1e-3:
                idx = np.searchsorted(x[:-1], mean[i], side="left")
                p = p + weight * np.where(x == x[idx], 1.0, 0.0)
            else:
                p = p + weight * np.exp(-0.5 * ((x - mean[i])/std[i])**2) / (np.sqrt(2*np.pi) * std[i])
        p = p / np.amax(p)     
        return p


def generate_random_points(means, errors, sample_size):
    points = np.tile(means, (sample_size,1))
    num_components = len(means)
    for j in range(num_components):
        if errors[j][0] == 0 and errors[j][1] == 0:
            points[:,j] = means[j] * np.ones((sample_size,))
        elif np.isnan(errors[j][0]) or np.isnan(errors[j][1]):
            points[:,j] = means[j] * np.ones((sample_size,))
        else:
            points[:,j] = np.random.uniform(means[j] + errors[j][0], means[j] + errors[j][1], size=(sample_size,))
    return points
        

if __name__ == '__main__':
    
    # Read out the parameters of the model
    filepath = get_filepath("Open the directory with the fitting results...")
    model_parameters, model_parameter_errors = load_model_parameters(filepath)
    
    # Compute the distance distribution
    distance_distribution = compute_distance_distribution(model_parameters, model_parameter_errors)
    
    # Find the optimal distance range
    indices_p_above_zero = np.where(distance_distribution['pu'] > 0.05*np.amax(distance_distribution['pu']))[0]
    r_low = distance_distribution['r'][np.amin(indices_p_above_zero)] - 5
    r_high = distance_distribution['r'][np.amax(indices_p_above_zero)] + 5
    if r_low < 15:
        r_low = 15
    if r_high > 80:
        r_high = 80
    #r_low, r_high = 25, 50
    
    # Save the distance distribution
    file = open(os.path.dirname(filepath)+'/distance_distribution.dat', 'w')
    file.write('{0:<20s}{1:<20s}{2:<20s}{3:<20s}{4:<20s}{5:<20s}{6:<20s}{7:<20s}\n'.format('r', 'p', 'p_lb', 'p_ub', 'p_lb(r_mean)', 'p_ub(r_width)', 'p_lb(r_mean)', 'p_ub(r_width)'))
    for i in range(distance_distribution['r'].size):
        file.write('{0:<20.3f}{1:<20.6f}{2:<20.6f}{3:<20.6f}{4:<20.6f}{5:<20.6f}{6:<20.6f}{7:<20.6f}\n'.format(distance_distribution['r'][i], 
                                                                                                               distance_distribution['p'][i], 
                                                                                                               distance_distribution['pl'][i], 
                                                                                                               distance_distribution['pu'][i],
                                                                                                               distance_distribution['pl_rm'][i], 
                                                                                                               distance_distribution['pu_rm'][i],
                                                                                                               distance_distribution['pl_rw'][i], 
                                                                                                               distance_distribution['pu_rw'][i],))
    file.close()

    # Plot the distance distribution
    fig = plt.figure(figsize=(7,6), facecolor='w', edgecolor='w')
    axes = fig.gca()
    axes.fill_between(distance_distribution['r'], distance_distribution['pu'], distance_distribution['pl'], color='limegreen', alpha=0.3, linewidth=0)    
    #axes.fill_between(distance_distribution['r'], distance_distribution['pu_rw'], distance_distribution['pl_rw'], color='green', alpha=0.5, linewidth=0)
    axes.fill_between(distance_distribution['r'], distance_distribution['pu_rm'], distance_distribution['pl_rm'], color='darkgreen', alpha=0.5, linewidth=0)
    axes.plot(distance_distribution['r'], distance_distribution['p'], color='black', linewidth=1.5)
    axes.set_xlim([r_low, r_high])
    axes.set_xlabel(r'$\mathit{r}$ ($\AA$)')
    axes.set_ylabel(r'$\mathit{P(r)}$ (a.u.)')
    fig.tight_layout()
    fig.savefig(os.path.dirname(filepath)+'/distance_distribution.png', format='png', dpi=600)
    plt.show()