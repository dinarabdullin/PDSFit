import numpy as np
from supplement.definitions import const


def save_symmetry_related_solutions(symmetry_related_solutions, parameters_indices, filepath):    
    ''' Saves symmetry-related sets of fitting parameters ''' 
    n_solutions = len(symmetry_related_solutions)
    file = open(filepath, 'w')
    file.write('{:<20}{:<20}{:<20}'.format('Parameter', 'No. spin pair', 'No. component',))
    for k in range(n_solutions):
        file.write('{:<20}'.format('Value('+symmetry_related_solutions[k]['transformation']+')'))
    file.write('\n')
    for parameter_name in const['angle_parameters_names']:
        parameter_indices = parameters_indices[parameter_name]
        for i in range(len(parameter_indices)):
            for j in range(len(parameter_indices[i])):
                file.write('{:<20}'.format(const['fitting_parameters_names_and_units'][parameter_name]))
                file.write('{:<20}'.format(i+1))
                file.write('{:<20}'.format(j+1))
                for k in range(n_solutions):
                    variable_value = symmetry_related_solutions[k]['variables'][parameter_name][i][j] / const['fitting_parameters_scales'][parameter_name]
                    file.write('{:<20}'.format(int(round(variable_value))))
                file.write('\n')
    file.write('\n')
    file.write('{:<60}'.format('score'))
    for k in range(n_solutions):
        score_value = symmetry_related_solutions[k]['score']
        file.write('{:<20.6f}'.format(score_value))
    file.write('\n')
    file.close()