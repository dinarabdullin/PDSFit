import sys
import numpy as np


def print_background_parameters(
    optimized_parameters, experiments, background, errors = []
    ):
    """Print the optimized and fixed parameters of a background model."""
    sys.stdout.write("\nBackground model parameters:\n")
    sys.stdout.write("{:<20}".format("Parameter"))
    sys.stdout.write("{:<15}".format("Experiment"))
    sys.stdout.write("{:<15}".format("Optimized"))
    sys.stdout.write("{:<15}".format("Value"))
    sys.stdout.write("{:<15}".format("-Error"))
    sys.stdout.write("{:<15}".format("+Error"))
    sys.stdout.write("\n")
    for name in background.parameter_names:
        for i in range(len(experiments)):
            sys.stdout.write("{:<20}".format(background.parameter_full_names[name]))
            sys.stdout.write("{:<15}".format(experiments[i].name))
            if background.parameters[name]["optimize"]:
                sys.stdout.write("{:<15}".format("yes"))
                value = optimized_parameters[i][name]
                if errors != []:
                    error = errors[i][name]
                else:
                    error = np.array([np.nan, np.nan])
            else:
                sys.stdout.write("{:<15}".format("no"))
                value = background.parameters[name]["value"]
                error = np.array([np.nan, np.nan])
            sys.stdout.write("{:<15.6f}".format(value)) 
            if not np.isnan(error[0]) and not np.isnan(error[1]):
                sys.stdout.write("{:<15.6f}{:<15.6f}".format(error[0], error[1])) 
            else:
                sys.stdout.write("{:<15}{:<15}".format("nan", "nan"))
            sys.stdout.write("\n")
    sys.stdout.flush()


def save_background_parameters(
    filepath, optimized_parameters, experiments, background, errors = []
    ):
    """Save background parameters and their errors."""
    file = open(filepath, "w")
    file.write("{:<20}".format("Parameter"))
    file.write("{:<15}".format("Experiment"))
    file.write("{:<15}".format("Optimized"))
    file.write("{:<15}".format("Value"))
    file.write("{:<15}".format("-Error"))
    file.write("{:<15}".format("+Error"))
    file.write("\n")
    for name in background.parameter_names:
        for i in range(len(experiments)):
            file.write("{:<20}".format(background.parameter_full_names[name]))
            file.write("{:<15}".format(experiments[i].name))
            if background.parameters[name]["optimize"]:
                file.write("{:<15}".format("yes"))
                value = optimized_parameters[i][name]
                if len(errors) != 0:
                    error = errors[i][name]
                else:
                    error = np.array([np.nan, np.nan])
            else:
                file.write("{:<15}".format("no"))
                value = background.parameters[name]["value"]
                error = np.array([np.nan, np.nan])
            file.write("{:<15.6f}".format(value)) 
            if not np.isnan(error[0]) and \
                not np.isnan(error[1]):
                file.write("{:<15.6f}{:<15.6f}".format(error[0], error[1])) 
            else:
                file.write("{:<15}{:<15}".format("nan", "nan"))
            file.write("\n")
    file.close()