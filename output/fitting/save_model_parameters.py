import sys
import numpy as np
from supplement.definitions import const


def print_model_parameters(optimized_parameters, fitting_parameters, errors = []):
    """Print the optimized and fixed parameters of a geometric model."""
    sys.stdout.write("\nGeometric model parameters:\n")
    sys.stdout.write("{:<20}".format("Parameter"))
    sys.stdout.write("{:<15}".format("No. component"))
    sys.stdout.write("{:<15}".format("Optimized"))
    sys.stdout.write("{:<15}".format("Value"))
    sys.stdout.write("{:<15}".format("-Error"))
    sys.stdout.write("{:<15}".format("+Error"))
    sys.stdout.write("\n")
    for name in const["model_parameter_names"]:
        for i, fitting_parameter in enumerate(fitting_parameters[name]):
            # Name
            sys.stdout.write("{:<20}".format(const["model_parameter_names_and_units"][name]))
            # No. component
            sys.stdout.write("{:<15}".format(i+1))
            # Optimization flag
            if fitting_parameter.is_optimized():
                sys.stdout.write("{:<15}".format("yes"))
            else:
                sys.stdout.write("{:<15}".format("no"))
            # Value
            if fitting_parameter.is_optimized():
                value = optimized_parameters[fitting_parameter.get_index()] / const["model_parameter_scales"][name]  
            else:
                value = fitting_parameter.get_value() / const["model_parameter_scales"][name]
            if name in const["angle_parameter_names"]:
                sys.stdout.write("{:<15.1f}".format(value))
            else:
                sys.stdout.write("{:<15.3f}".format(value))
            # -Error and +Error
            if fitting_parameter.is_optimized():
                if errors != []:
                    error = errors[fitting_parameter.get_index()]
                    if not np.isnan(error[0]) and not np.isnan(error[1]):
                        error = [v / const["model_parameter_scales"][name] for v in error]
                        if const["paired_model_parameters"][name] != "none":
                            paired_name = const["paired_model_parameters"][name]
                            paired_fitting_parameter = fitting_parameters[paired_name][i]
                            if paired_fitting_parameter.is_optimized():
                                paired_error = errors[paired_fitting_parameter.get_index()]
                                if np.isnan(paired_error[0]) or np.isnan(paired_error[1]):
                                    sys.stdout.write("{:<15}{:<15}".format("nan", "nan"))
                                else:
                                    if name in const["angle_parameter_names"]:
                                        sys.stdout.write("{:<15.1f}{:<15.1f}".format(error[0], error[1]))
                                    else:
                                        sys.stdout.write("{:<15.3f}{:<15.3f}".format(error[0], error[1]))
                            else:
                                if name in const["angle_parameter_names"]:
                                    sys.stdout.write("{:<15.1f}{:<15.1f}".format(error[0], error[1]))
                                else:
                                    sys.stdout.write("{:<15.3f}{:<15.3f}".format(error[0], error[1]))  
                        else:
                            if name in const["angle_parameter_names"]:
                                sys.stdout.write("{:<15.1f}{:<15.1f}".format(error[0], error[1]))
                            else:
                                sys.stdout.write("{:<15.3f}{:<15.3f}".format(error[0], error[1]))
                    else:
                        sys.stdout.write("{:<15}{:<15}".format("nan", "nan")) 
                else:
                    sys.stdout.write("{:<15}{:<15}".format("nan", "nan")) 
            else:
                sys.stdout.write("{:<15}{:<15}".format("nan", "nan")) 
            sys.stdout.write("\n")
    sys.stdout.flush()


def save_model_parameters(filepath, optimized_parameters, fitting_parameters, errors = []):    
    """Save the optimized and fixed parameters of a geometric model."""
    file = open(filepath, "w")
    file.write("{:<20}".format("Parameter"))
    file.write("{:<15}".format("No. component"))
    file.write("{:<15}".format("Optimized"))
    file.write("{:<15}".format("Value"))
    file.write("{:<15}".format("-Error"))
    file.write("{:<15}".format("+Error"))
    file.write("\n")
    for name in const["model_parameter_names"]:
        for i, fitting_parameter in enumerate(fitting_parameters[name]):
            # Name
            file.write("{:<20}".format(const["model_parameter_names_and_units"][name]))
            # No. component
            file.write("{:<15}".format(i+1))
            # Optimization flag
            if fitting_parameter.is_optimized():
                file.write("{:<15}".format("yes"))
            else:
                file.write("{:<15}".format("no"))
            # Value
            if fitting_parameter.is_optimized():
                value = optimized_parameters[fitting_parameter.get_index()] / const["model_parameter_scales"][name]  
            else:
                value = fitting_parameter.get_value() / const["model_parameter_scales"][name]
            if name in const["angle_parameter_names"]:
                file.write("{:<15.1f}".format(value))
            else:
                file.write("{:<15.3f}".format(value))
            # -Error and +Error
            if fitting_parameter.is_optimized():
                if errors != []:
                    error = errors[fitting_parameter.get_index()]
                    if not np.isnan(error[0]) and not np.isnan(error[1]):
                        error = [v / const["model_parameter_scales"][name] for v in error]
                        if const["paired_model_parameters"][name] != "none":
                            paired_name = const["paired_model_parameters"][name]
                            paired_fitting_parameter = fitting_parameters[paired_name][i]
                            if paired_fitting_parameter.is_optimized():
                                paired_error = errors[paired_fitting_parameter.get_index()]
                                if np.isnan(paired_error[0]) or np.isnan(paired_error[1]):
                                    file.write("{:<15}{:<15}".format("nan", "nan"))
                                else:
                                    if name in const["angle_parameter_names"]:
                                        file.write("{:<15.1f}{:<15.1f}".format(error[0], error[1]))
                                    else:
                                        file.write("{:<15.3f}{:<15.3f}".format(error[0], error[1]))
                            else:
                                if name in const["angle_parameter_names"]:
                                    file.write("{:<15.1f}{:<15.1f}".format(error[0], error[1]))
                                else:
                                    file.write("{:<15.3f}{:<15.3f}".format(error[0], error[1]))  
                        else:
                            if name in const["angle_parameter_names"]:
                                file.write("{:<15.1f}{:<15.1f}".format(error[0], error[1]))
                            else:
                                file.write("{:<15.3f}{:<15.3f}".format(error[0], error[1]))
                    else:
                        file.write("{:<15}{:<15}".format("nan", "nan")) 
                else:
                    file.write("{:<15}{:<15}".format("nan", "nan")) 
            else:
                file.write("{:<15}{:<15}".format("nan", "nan")) 
            file.write("\n")
    file.close()


def save_model_parameters_all_runs(filepath, optimized_parameters_all_runs, fitting_parameters):    
    """Save the optimized and fixed parameters of several geometric models."""
    file = open(filepath, "w")
    file.write("{:<20}".format("Parameter"))
    file.write("{:<15}".format("No. component"))
    file.write("{:<15}".format("Optimized"))
    num_runs = len(optimized_parameters_all_runs)
    for r in range(num_runs):
        file.write("{:<15}".format("Run" + str(r + 1)))
    file.write("\n")
    for name in const["model_parameter_names"]:
        for i, fitting_parameter in enumerate(fitting_parameters[name]):
            # Name
            file.write("{:<20}".format(const["model_parameter_names_and_units"][name]))
            # No. component
            file.write("{:<15}".format(i+1))
            # Optimization flag
            if fitting_parameter.is_optimized():
                file.write("{:<15}".format("yes"))
            else:
                file.write("{:<15}".format("no"))
            # Values
            if fitting_parameter.is_optimized():
                for r in range(num_runs):
                    value = optimized_parameters_all_runs[r][fitting_parameter.get_index()] / const["model_parameter_scales"][name]
                    if name in const["angle_parameter_names"]:
                        file.write("{:<15.1f}".format(value))
                    else:
                        file.write("{:<15.3f}".format(value))
            else:
                value = fitting_parameter.get_value() / const["model_parameter_scales"][name]
                for r in range(num_runs):
                    if name in const["angle_parameter_names"]:
                        file.write("{:<15.1f}".format(value))
                    else:
                        file.write("{:<15.3f}".format(value))
            file.write("\n") 
    file.close()