import numpy as np
from supplement.definitions import const


def save_score_vs_parameters(score_vs_parameter_subset, error_analysis_parameters, filepath):
    ''' Saves the score as a function of a fitting parameters' subset ''' 
    file = open(filepath, 'w')
    n_par = len(error_analysis_parameters)
    for j in range(n_par):
        parameter_id = error_analysis_parameters[j]
        name = parameter_id.name
        spin_pair = parameter_id.spin_pair
        component = parameter_id.component
        text = name + ' ' + str(spin_pair+1) + ' ' + str(component+1) 
        file.write('{:<20}'.format(text))
    file.write('{:<20}\n'.format('chi2'))
    n_points = len(score_vs_parameter_subset['score'])
    for i in range(n_points):
        for j in range(n_par):
            parameter_id = error_analysis_parameters[j]
            parameter_value = score_vs_parameter_subset['parameters'][j][i] / const['fitting_parameters_scales'][parameter_id.name]
            file.write('{:<20.4f}'.format(parameter_value))
        score_value = score_vs_parameter_subset['score'][i]
        file.write('{:<20.6f}\n'.format(score_value))
    file.close()