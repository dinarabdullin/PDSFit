import numpy as np
from supplement.definitions import const


def save_error_surface(filepath, error_surface):
    """Save an error surface.""" 
    parameters, parameter_grid_points, chi2_values = error_surface["par"], error_surface["x"], error_surface["y"]
    num_parameters = len(parameters)
    file = open(filepath, "w")
    for i in range(num_parameters):
        parameter = parameters[i]
        name, component = parameter.name, parameter.component
        column_name = const["model_parameter_names_and_units"][name] + ", comp. {0}".format(component + 1)
        file.write("{:<30}".format(column_name))
    file.write("{:<30}\n".format("chi-squared"))
    for j in range(parameter_grid_points.shape[1]):
        for i in range(num_parameters):
            parameter = parameters[i]
            name = parameter.name
            parameter_value = parameter_grid_points[i][j] / const["model_parameter_scales"][name]
            if name in const["angle_parameter_names"]:
                file.write("{:<30.1f}".format(parameter_value))
            else:
                file.write("{:<30.3f}".format(parameter_value))
        chi2_value = chi2_values[j]
        file.write("{:<30.1f}\n".format(chi2_value))
    file.close()