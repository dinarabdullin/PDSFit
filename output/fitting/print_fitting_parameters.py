import sys
import numpy as np
from supplement.definitions import const


def print_fitting_parameters(parameters_indices, optimized_parameters_values, fixed_parameters_values, parameter_errors=[]):
    ''' Prints optimized and fixed fitting parameters as a table '''
    sys.stdout.write('\nOptimized fitting parameters:\n')
    sys.stdout.write('{:<20}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}\n'.format('Parameter', 'No. spin pair', 'No. component', 'Optimized', 'Value', '-Error', '+Error'))
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
                    if parameter_errors != []:
                        if not np.isnan(parameter_errors[parameter_object.index][0]) and not np.isnan(parameter_errors[parameter_object.index][1]):
                            variable_error = parameter_errors[parameter_object.index] / const['fitting_parameters_scales'][parameter_name]
                            if const['paired_fitting_parameters'][parameter_name] != 'none':
                                paired_parameter_name = const['paired_fitting_parameters'][parameter_name]
                                paired_parameter_object = parameters_indices[paired_parameter_name][i][j]
                                if paired_parameter_object.optimize:
                                    if np.isnan(parameter_errors[paired_parameter_object.index][0]) or np.isnan(parameter_errors[paired_parameter_object.index][1]):
                                        sys.stdout.write('{:<15}{:<15}'.format('nan', 'nan'))
                                    else:
                                        sys.stdout.write('{:<15.4}{:<15.4}'.format(variable_error[0], variable_error[1]))
                                else:
                                    sys.stdout.write('{:<15.4}{:<15.4}'.format(variable_error[0], variable_error[1]))
                            else:
                                sys.stdout.write('{:<15.4}{:<15.4}'.format(variable_error[0], variable_error[1]))
                        else:
                            sys.stdout.write('{:<15}{:<15}'.format('nan', 'nan'))
                    else:
                        sys.stdout.write('{:<15}{:<15}'.format('nan', 'nan'))
                else:
                    sys.stdout.write('{:<15}{:<15}'.format('nan', 'nan')) 
                sys.stdout.write('\n')