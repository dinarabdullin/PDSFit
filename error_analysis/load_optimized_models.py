import os
import sys
import numpy as np
from fitting.parameter_id import ParameterID
from supplement.definitions import const


def load_optimized_model(filepath, model_only = False):
    """Load the optimized (or fixed) values of model parameters from file 
    fitting_parameters.dat."""
    # Prepare containers
    fitting_parameters = {}
    for name in const["model_parameter_names"]:
        fitting_parameters[name] = []
    fitting_index = 0
    optimized_model, errors = [], []
    # Read the file
    file = open(filepath, "r")
    next(file)
    for line in file:
        first_column = list(chunk_string(line[0:19], 20))
        next_columns = list(chunk_string(line[20:-1], 15))
        all_columns = first_column + next_columns
        data = [v.strip() for v in all_columns]
        longname = data[0]
        name_found = False
        for key in const["model_parameter_names_and_units"]:    
            if longname == const["model_parameter_names_and_units"][key]:
                name = key
                name_found = True
        if not name_found:
            raise ValueError("Invalid parameter name \'{0}\'!".format(name))
            sys.exit(1)
        component = int(data[1]) - 1
        optimized = bool(data[2])
        value = float(data[3]) * const["model_parameter_scales"][name]
        minus_error, plus_error = data[4], data[5]
        if minus_error == "nan" or plus_error == "nan":
            error = np.array([np.nan, np.nan])
        else:
            error = np.array([
                float(minus_error) * const["model_parameter_scales"][name],
                float(plus_error) * const["model_parameter_scales"][name]
                ])
        # Store the loaded data
        fitting_parameter = ParameterID(name, component)
        fitting_parameter.set_optimized(optimized)
        fitting_parameter.set_index(fitting_index)
        fitting_parameter.set_value(value)
        fitting_parameters[name].append(fitting_parameter)
        if optimized:
            optimized_model.append(value)
            errors.append(error)
            fitting_index += 1
    if model_only:
        return optimized_model
    else:
        return fitting_parameters, optimized_model, errors


def chunk_string(string, length):
    """Chunk string based on the chunk length(s)."""
    return (string[0+i:length+i] for i in range(0, len(string), length))


def load_fitting_parameters(filepath):
    """Load fitting parameters from file fitting_parameters.dat."""
    fitting_parameters, _1, _2 = load_optimized_model(filepath)
    return fitting_parameters


def load_optimized_models(filepath):
    """Load the optimized (or fixed) values of model parameters from files
    fitting_parameters.dat and fitting_parameters_run{i}.dat, where i is the
    number of an optimization run."""
    _, optimized_model, errors = load_optimized_model(filepath)
    c = 1
    optimized_models = []
    while True:
        new_filepath = filepath[:-4] + "_run" + str(c) + ".dat"
        if os.path.exists(new_filepath):
            model = load_optimized_model(new_filepath, model_only = True)
            optimized_models.append(model)
        else:
            break
        c += 1
    return optimized_model, errors, optimized_models