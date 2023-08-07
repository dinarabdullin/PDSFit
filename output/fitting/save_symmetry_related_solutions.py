import numpy as np
from supplement.definitions import const


def save_symmetry_related_solutions(symmetry_related_solutions, fitting_parameters, filepath):    
    ''' Saves symmetry-related sets of model parameters ''' 
    n_solutions = len(symmetry_related_solutions)
    file = open(filepath, 'w')
    file.write('{:<20}{:<15}'.format('Parameter', 'No. component'))
    for k in range(n_solutions):
        file.write('{:<15}'.format(symmetry_related_solutions[k]['transformation']))
    file.write('\n')
    for parameter_name in const['model_parameter_names']:
        parameter_indices = fitting_parameters['indices'][parameter_name]
        for i in range(len(parameter_indices)):
            file.write('{:<20}'.format(const['model_parameter_names_and_units'][parameter_name]))
            file.write('{:<15}'.format(i+1))
            for k in range(n_solutions):
                variable_value = symmetry_related_solutions[k]['variables'][parameter_name][i] / const['model_parameter_scales'][parameter_name]
                file.write('{:<15.1f}'.format(variable_value))
            file.write('\n')
    file.write('\n')
    file.write('{:<35}'.format('score'))
    for k in range(n_solutions):
        score_value = symmetry_related_solutions[k]['score']
        file.write('{:<15.1f}'.format(score_value))
    file.write('\n')
    file.close()