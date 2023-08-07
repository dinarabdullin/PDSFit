import os
import sys
import numpy as np
from supplement.definitions import const


def export_optimized_model_parameters(filepath):
    # Read file
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
        loaded_parameter['value'] = float(data[3]) * const['model_parameter_scales'][loaded_parameter['name']]
        minus_error = data[4].strip()
        plus_error = data[5].strip()
        if minus_error == 'nan' or plus_error == 'nan':
            minus_error_value = np.nan
            plus_error_value = np.nan
        else:
            minus_error_value = float(minus_error) * const['model_parameter_scales'][loaded_parameter['name']]
            plus_error_value = float(plus_error) * const['model_parameter_scales'][loaded_parameter['name']]          
        loaded_parameter['errors'] = np.array([minus_error_value, plus_error_value])
        loaded_parameters.append(loaded_parameter)
    return loaded_parameters


def load_optimized_model_parameters(filepath):
    ''' Loads the optimized values of model parameters '''
    # One set of optimized parameters
    loaded_parameters = export_optimized_model_parameters(filepath)
    model_parameters = []
    model_parameter_errors = []
    for parameter in loaded_parameters:
        if parameter['optimized']:
            model_parameters.append(parameter['value'])
            model_parameter_errors.append(parameter['errors'])
    model_parameters = np.array(model_parameters)
    model_parameter_errors = np.array(model_parameter_errors)
    # All sets of optimized model parameters
    model_parameters_all_runs = []
    c = 1
    while True:
        filename = filepath[:-4] + '_run' + str(c) + '.dat'
        c += 1
        if os.path.exists(filename):
            loaded_parameters = export_optimized_model_parameters(filename)
            model_parameters_single_run = []
            for parameter in loaded_parameters:
                if parameter['optimized']:
                    model_parameters_single_run.append(parameter['value'])
            model_parameters_single_run = np.array(model_parameters_single_run)
            model_parameters_all_runs.append(model_parameters_single_run)
        else:
            break
    return model_parameters, model_parameter_errors, model_parameters_all_runs


def chunk_string(string, length):
    return (string[0+i:length+i] for i in range(0, len(string), length))