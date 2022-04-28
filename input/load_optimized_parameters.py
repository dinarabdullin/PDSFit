import os
import sys
import numpy as np
from supplement.definitions import const


def load_optimized_parameters(filepath):
    ''' Reads out optimized values of fitting parameters from a file '''
    # Read file
    loaded_parameters = [] 
    file = open(filepath, 'r')
    next(file)
    for line in file:
        data = list(chunk_string(line, 20))
        loaded_parameter = {}
        name = data[0].strip()
        name_found = False
        for key in const['fitting_parameters_names_and_units']:    
            if name == const['fitting_parameters_names_and_units'][key]:
                loaded_parameter['name'] = key
                name_found = True
        if not name_found:
            raise ValueError('Error is found in the file with the optimized fitting parameters!')
            sys.exit(1)
        loaded_parameter['spin_pair'] = int(data[1])
        loaded_parameter['component'] = int(data[2])
        optimized = data[3].strip()
        if optimized == 'yes':
            loaded_parameter['optimized'] = 1
        elif optimized == 'no':
            loaded_parameter['optimized'] = 0
        else:
            print('Error is found in the file with the optimized fitting parameters!')
        loaded_parameter['value'] = float(data[4]) * const['fitting_parameters_scales'][loaded_parameter['name']]
        minus_error = data[5].strip()
        plus_error = data[6].strip()
        if minus_error == 'nan' or plus_error == 'nan':
            minus_error_value = np.nan
            plus_error_value = np.nan
        else:
            minus_error_value = float(minus_error) * const['fitting_parameters_scales'][loaded_parameter['name']]
            plus_error_value = float(plus_error) * const['fitting_parameters_scales'][loaded_parameter['name']]          
        loaded_parameter['errors'] = np.array([minus_error_value, plus_error_value])
        loaded_parameters.append(loaded_parameter)
    # Extract the optimized values of fitting parameters
    optimized_parameters = []
    parameter_error = []
    for parameter in loaded_parameters:
        if parameter['optimized']:
            optimized_parameters.append(parameter['value'])
            parameter_error.append(parameter['errors'])
    return np.array(optimized_parameters), np.array(parameter_error)


def chunk_string(string, length):
    return (string[0+i:length+i] for i in range(0, len(string), length))