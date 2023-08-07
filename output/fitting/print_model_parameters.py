import sys
import numpy as np
from supplement.definitions import const


def print_model_parameters(optimized_model_parameters, model_parameter_errors, fitting_parameters):
    ''' Prints the optimized and fixed parameters of the model '''
    sys.stdout.write('\nModel parameters:\n')
    sys.stdout.write('{:<20}{:<15}{:<15}{:<15}{:<15}{:<15}\n'.format('Parameter', 'No. component', 'Optimized', 'Value', '-Error', '+Error'))
    for parameter_name in const['model_parameter_names']:
        parameter_indices = fitting_parameters['indices'][parameter_name]
        for i in range(len(parameter_indices)):
            sys.stdout.write('{:<20}'.format(const['model_parameter_names_and_units'][parameter_name]))
            sys.stdout.write('{:<15}'.format(i+1))
            parameter_object = parameter_indices[i]
            if parameter_object.optimize:
                sys.stdout.write('{:<15}'.format('yes'))
            else:
                sys.stdout.write('{:<15}'.format('no'))
            if parameter_object.optimize:
                parameter_value = optimized_model_parameters[parameter_object.index] / const['model_parameter_scales'][parameter_name]
                if parameter_name in const['angle_parameter_names']:
                    sys.stdout.write('{:<15.1f}'.format(parameter_value))
                else:
                    sys.stdout.write('{:<15.3f}'.format(parameter_value))
            else:
                parameter_value = fitting_parameters['values'][parameter_object.index]  / const['model_parameter_scales'][parameter_name]
                if parameter_name in const['angle_parameter_names']:
                    sys.stdout.write('{:<15.1f}'.format(parameter_value))
                else:
                    sys.stdout.write('{:<15.3f}'.format(parameter_value))
            if parameter_object.optimize:
                if model_parameter_errors != []:
                    if not np.isnan(model_parameter_errors[parameter_object.index][0]) and not np.isnan(model_parameter_errors[parameter_object.index][1]):
                        parameter_error = model_parameter_errors[parameter_object.index] / const['model_parameter_scales'][parameter_name]
                        if const['paired_model_parameters'][parameter_name] != 'none':
                            paired_parameter_name = const['paired_model_parameters'][parameter_name]
                            paired_parameter_object = fitting_parameters['indices'][paired_parameter_name][i]
                            if paired_parameter_object.optimize:
                                if np.isnan(model_parameter_errors[paired_parameter_object.index][0]) or np.isnan(model_parameter_errors[paired_parameter_object.index][1]):
                                    sys.stdout.write('{:<15}{:<15}'.format('nan', 'nan'))
                                else:
                                    if parameter_name in const['angle_parameter_names']:
                                        sys.stdout.write('{:<15.1f}{:<15.1f}'.format(parameter_error[0], parameter_error[1]))
                                    else:
                                        sys.stdout.write('{:<15.3f}{:<15.3f}'.format(parameter_error[0], parameter_error[1]))
                            else:
                                if parameter_name in const['angle_parameter_names']:
                                    sys.stdout.write('{:<15.1f}{:<15.1f}'.format(parameter_error[0], parameter_error[1]))
                                else:
                                    sys.stdout.write('{:<15.3f}{:<15.3f}'.format(parameter_error[0], parameter_error[1]))
                        else:
                            if parameter_name in const['angle_parameter_names']:
                                sys.stdout.write('{:<15.1f}{:<15.1f}'.format(parameter_error[0], parameter_error[1]))
                            else:
                                sys.stdout.write('{:<15.3f}{:<15.3f}'.format(parameter_error[0], parameter_error[1]))
                    else:
                        sys.stdout.write('{:<15}{:<15}'.format('nan', 'nan'))
                else:
                    sys.stdout.write('{:<15}{:<15}'.format('nan', 'nan'))
            else:
                sys.stdout.write('{:<15}{:<15}'.format('nan', 'nan')) 
            sys.stdout.write('\n')
    sys.stdout.flush()