import numpy as np
from supplement.definitions import const


def save_error_profile(error_profile, parameter_id, filepath):
    ''' Saves chi-squared as a function of a fitting parameters' subset ''' 
    file = open(filepath, 'w')
    text = parameter_id.name + ' ' + str(parameter_id.component+1) 
    file.write('{:<20}'.format(text))
    file.write('{:<20}\n'.format('chi2'))
    n_points = len(error_profile['chi2'])
    for i in range(n_points):
        parameter_value = error_profile['parameter'][i] / const['model_parameter_scales'][parameter_id.name]
        if parameter_id.name in const['angle_parameter_names']:
            file.write('{:<20.1f}'.format(parameter_value))
        else:
            file.write('{:<20.3f}'.format(parameter_value))
        chi2_value = error_profile['chi2'][i]
        file.write('{:<20.1f}\n'.format(chi2_value))
    file.close()