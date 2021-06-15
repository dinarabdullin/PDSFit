import sys
import numpy as np
from supplement.definitions import const


def print_fitting_parameters(parameters_indices, optimized_parameters_values, fixed_parameters_values, parameters_errors=[]):
    ''' Displays the optimized and fixed fitting parameters into a single dictionary '''
    sys.stdout.write('\nOptimized fitting parameters:\n')
    sys.stdout.write('{:<20}{:<15}{:<15}{:<15}{:<15}{:<15}\n'.format('Parameter', 'No. spin pair', 'No. component', 'Optimized', 'Value', 'Precision'))
    for parameter_name in const['fitting_parameters_names']:
        parameter_indices = parameters_indices[parameter_name]
        for i in range(len(parameter_indices)):
            for j in range(len(parameter_indices[i])):
                sys.stdout.write('{:<20}'.format(const['fitting_parameters_names_and_units'][parameter_name]))
                sys.stdout.write('{:<15}'.format(i+1))
                sys.stdout.write('{:<15}'.format(j+1))
                parameter_object = parameter_indices[i][j]
                if parameter_object.optimize:
                    sys.stdout.write('{:<15}'.format('yes'))
                else:
                    sys.stdout.write('{:<15}'.format('no'))
                if parameter_object.optimize:
                    variable_value = optimized_parameters_values[parameter_object.index] / const['fitting_parameters_scales'][parameter_name]
                    sys.stdout.write('{:<15.4}'.format(variable_value))
                else:
                    variable_value = fixed_parameters_values[parameter_object.index]  / const['fitting_parameters_scales'][parameter_name]
                    sys.stdout.write('{:<15.4}'.format(variable_value))
                if parameter_object.optimize:
                    if parameters_errors != []:
                        if not np.isnan(parameters_errors[parameter_object.index]):
                            variable_error = parameters_errors[parameter_object.index] / const['fitting_parameters_scales'][parameter_name]
                            sys.stdout.write('{:<15.4}'.format(variable_error))
                        else:
                            sys.stdout.write('{:<15}'.format('nan'))
                    else:
                        sys.stdout.write('{:<15}'.format('nan'))
                else:
                    sys.stdout.write('{:<15}'.format('nan'))    
                sys.stdout.write('\n')