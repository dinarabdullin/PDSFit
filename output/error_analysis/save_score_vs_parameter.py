import numpy as np
from supplement.definitions import const


def save_score_vs_parameter(score_vs_parameter, parameter_id, filepath):
    ''' Saves the score as a function of a fitting parameters' subset ''' 
    file = open(filepath, 'w')
    name = parameter_id.name
    spin_pair = parameter_id.spin_pair
    component = parameter_id.component
    text = name + ' ' + str(spin_pair+1) + ' ' + str(component+1) 
    file.write('{:<20}'.format(text))
    file.write('{:<20}\n'.format('chi2'))
    n_points = len(score_vs_parameter['score'])
    for i in range(n_points):
        parameter_value = score_vs_parameter['parameter'][i] / const['fitting_parameters_scales'][name]
        file.write('{:<20.4f}'.format(parameter_value))
        score_value = score_vs_parameter['score'][i]
        file.write('{:<20.6f}\n'.format(score_value))
    file.close()