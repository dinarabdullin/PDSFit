import numpy as np
from supplement.definitions import const


def save_error_surface(error_surface, error_analysis_parameters, filepath):
    ''' Saves chi2 as a function of a fitting parameter subset ''' 
    file = open(filepath, 'w')
    n_par = len(error_analysis_parameters)
    for j in range(n_par):
        parameter_id = error_analysis_parameters[j]
        name = parameter_id.name
        component = parameter_id.component
        text = name + ' ' + str(component+1) 
        file.write('{:<20}'.format(text))
    file.write('{:<20}\n'.format('chi2'))
    n_points = len(error_surface['chi2'])
    for i in range(n_points):
        for j in range(n_par):
            parameter_id = error_analysis_parameters[j]
            parameter_value = error_surface['parameters'][j][i] / const['model_parameter_scales'][parameter_id.name]
            if parameter_id.name in const['angle_parameter_names']:
                file.write('{:<20.1f}'.format(parameter_value))
            else:
                file.write('{:<20.3f}'.format(parameter_value))
        chi2_value = error_surface['chi2'][i]
        file.write('{:<20.1f}\n'.format(chi2_value))
    file.close()