import os
import io
import wx
import numpy as np
from textwrap import wrap
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


def load_error_analysis_data(filepath):
    file = open(filepath, 'r')
    next(file)
    r_mean_values, r_width_values, chi2_values = [], [], []
    for line in file:
        data = wrap(line, 20)
        r_mean_values.append(float(data[0]))
        r_width_values.append(float(data[1]))
        chi2_values.append(float(data[2]))
    error_surface = {}
    error_surface['r_mean'] = np.array(r_mean_values)
    error_surface['r_width'] = np.array(r_width_values)
    error_surface['chi2'] = np.array(chi2_values)
    print(error_surface)
    file.close()
    return error_surface


def chunk_string(string, length):
    return (string[0+i:length+i] for i in range(0, len(string), length))


def compute_distance_distribution(model_parameters, error_surface, chi2_threshold, chi2_minimum, r_minimum=15, r_maximum=80, r_increment=0.01):
    distance_distribution = {}
    distance_distribution['r'] = np.arange(r_minimum, r_maximum+r_increment, r_increment)
    distance_distribution['p'] = multimodal_normal_distribution(distance_distribution['r'], model_parameters['r_mean'], model_parameters['r_width'], model_parameters['rel_prob'])
    distance_distribution['pl'] = copy.deepcopy(distance_distribution['p'])
    distance_distribution['pu'] = copy.deepcopy(distance_distribution['p'])
    print(model_parameters)
    
    
    n_components = len(model_parameters['r_mean'])
    if n_components == 1:
        r_mean_values = error_surface['r_mean'] 
        r_width_values = error_surface['r_width'] 
        chi2_values = error_surface['chi2']
        selected_indices = np.where(chi2_values <= chi2_minimum + chi2_threshold)[0]
        selected_r_mean_values = r_mean_values[selected_indices]
        selected_r_width_values = r_width_values[selected_indices]
        rel_prob_values = model_parameters['rel_prob'] * np.ones(selected_indices.size)
    else: 
        print('Error: Only unimodal distribution is supported!')
    
    n_samples = len(selected_r_mean_values)
    print(n_samples)
    for i in range(n_samples):
        test_distance_distribution = multimodal_normal_distribution(distance_distribution['r'], [selected_r_mean_values[i]], [selected_r_width_values[i]], [rel_prob_values[i]])
        distance_distribution['pl'] = np.amin(np.array([distance_distribution['pl'], test_distance_distribution]), axis=0)
        distance_distribution['pu'] = np.amax(np.array([distance_distribution['pu'], test_distance_distribution]), axis=0)
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


def load_chi2_threshold(filepath):
    file = open(filepath, 'r')
    lines = file.readlines()
    for line in lines:
        if line.find('Total chi-squared threshold') != -1:
            content = list(line.split())
            chi2_threshold = float(content[4])      
    file.close()
    
    file = open(filepath, 'r')
    lines = file.readlines()
    for line in lines:
        if line.find('Minimum chi-squared:') != -1:
            content = list(line.split())
            chi2_minimum = float(content[2])      
    file.close()
    
    print('\nChi-squared threshold: {0}'.format(chi2_threshold))
    print('\nMinimum chi-squared: {0}'.format(chi2_minimum))
    return chi2_threshold, chi2_minimum
    

if __name__ == '__main__':
    
    # Read out the parameters of the model
    filepath1 = get_filepath("Open the file with the fitting results...")
    directory = os.path.dirname(filepath1)+'/'
    model_parameters, model_parameter_errors = load_model_parameters(filepath1)
    
    # Error surface
    error_surface = load_error_analysis_data(directory + 'error_surface_1.dat')
    
    # Score threshold
    chi2_threshold, chi2_minimum = load_chi2_threshold(directory + 'logfile.log')

    # Compute the distance distribution
    distance_distribution = compute_distance_distribution(model_parameters, error_surface, chi2_threshold, chi2_minimum)

    # Find the optimal distance range
    indices_p_above_zero = np.where(distance_distribution['pu'] > 0.05*np.amax(distance_distribution['pu']))[0]
    r_low = distance_distribution['r'][np.amin(indices_p_above_zero)] - 5
    r_high = distance_distribution['r'][np.amax(indices_p_above_zero)] + 5
    # if r_low < 15:
        # r_low = 15
    # if r_high > 80:
        # r_high = 80
    r_low, r_high = 25, 50
    
    # Save the distance distribution
    file = open(os.path.dirname(filepath1)+'/distance_distribution.dat', 'w')
    for i in range(distance_distribution['r'].size):
        file.write('{0:<20.3f}{1:<20.6f}{2:<20.6f}{3:<20.6f}\n'.format(distance_distribution['r'][i], distance_distribution['p'][i], distance_distribution['pl'][i], distance_distribution['pu'][i]))
    file.close()

    # Plot the distance distribution
    fig = plt.figure(figsize=(7,6), facecolor='w', edgecolor='w')
    axes = fig.gca()
    axes.fill_between(distance_distribution['r'], distance_distribution['pu'], distance_distribution['pl'], color='black', alpha=0.3, linewidth=0)
    axes.plot(distance_distribution['r'], distance_distribution['p'], color='black', linewidth=1.5)
    axes.set_xlim([r_low, r_high])
    axes.set_xlabel(r'$\mathit{r}$ ($\AA$)')
    axes.set_ylabel(r'$\mathit{P(r)}$ (a.u.)')
    fig.tight_layout()
    fig.savefig(os.path.dirname(filepath1)+'/distance_distribution.png', format='png', dpi=600)
    plt.show()