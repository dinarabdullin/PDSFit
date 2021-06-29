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
        print(name, len(name))
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
        precision = data[5].strip()
        if precision == 'nan':
            loaded_parameter['precision'] = np.nan
        else:
            loaded_parameter['precision'] = float(precision) * const['fitting_parameters_scales'][loaded_parameter['name']]
        loaded_parameters.append(loaded_parameter)
    # Extract the optimized values of fitting parameters
    optimized_parameters = []
    parameter_errors = []
    for parameter in loaded_parameters:
        if parameter['optimized']:
            optimized_parameters.append(parameter['value'])
            parameter_errors.append(parameter['precision'])
    return np.array(optimized_parameters), np.array(parameter_errors)


def chunk_string(string, length):
    return (string[0+i:length+i] for i in range(0, len(string), length))