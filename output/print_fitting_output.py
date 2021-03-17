import sys
from supplement.definitions import const


def print_fitting_parameters(parameters_indices, optimized_parameters_values, fixed_parameters_values, optimized_parameters_errors=[]):
    ''' Displays optimized and fixed fitting parameters into a single dictionary '''
    sys.stdout.write('\nOptimized fitting parameters:\n')
    sys.stdout.write("{:<20}{:<15}{:<15}{:<15}{:<15}{:<15}\n".format('Parameter', 'No. spin pair', 'No. component', 'Optimized', 'Value', 'Precision'))
    for variable_name in const['fitting_parameters_names']:
        variable_indices = parameters_indices[variable_name]
        for i in range(len(variable_indices)):
            for j in range(len(variable_indices[i])):
                sys.stdout.write('{:<20}'.format(const['fitting_parameters_names_and_units'][variable_name]))
                sys.stdout.write('{:<15}'.format(i+1))
                sys.stdout.write('{:<15}'.format(j+1))
                variable_id = variable_indices[i][j]
                if variable_id.opt:
                    sys.stdout.write('{:<15}'.format('yes'))
                else:
                    sys.stdout.write('{:<15}'.format('no'))
                if variable_id.opt:
                    variable_value = optimized_parameters_values[variable_id.idx] / const['fitting_parameters_scales'][variable_name]
                    sys.stdout.write('{:<15.4}'.format(variable_value))
                else:
                    variable_value = fixed_parameters_values[variable_id.idx]  / const['fitting_parameters_scales'][variable_name]
                    sys.stdout.write('{:<15.4}'.format(variable_value))
                if variable_id.opt:
                    if optimized_parameters_errors != []:
                        if optimized_parameters_errors[variable_id.idx] != 0:
                            variable_error = optimized_parameters_errors[variable_id.idx] / const['fitting_parameters_scales'][variable_name]
                            sys.stdout.write('{:<15.4}'.format(variable_error))
                        else:
                            sys.stdout.write('{:<15}'.format('nan'))
                    else:
                        sys.stdout.write('{:<15}'.format('nan'))
                else:
                    sys.stdout.write('{:<15}'.format('nan'))    
                sys.stdout.write('\n')


def print_modulation_depth_scale_factors(modulation_depth_scale_factors, experiments):
    ''' Displays the scale factors of modulation depths '''
    sys.stdout.write('\nScale factors of modulation depths:\n')
    sys.stdout.write("{:<20}{:<15}\n".format('Experiment', 'Scale factor'))
    for i in range(len(experiments)):
        sys.stdout.write('{:<20}'.format(experiments[i].name))
        sys.stdout.write('{:<15.4}'.format(modulation_depth_scale_factors[i]))
        sys.stdout.write('\n')