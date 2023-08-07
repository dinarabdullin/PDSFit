import numpy as np
from supplement.definitions import const


def save_model_parameters(optimized_model_parameters, model_parameter_errors, fitting_parameters, filepath):    
    ''' Saves the model parameters and their errors ''' 
    file = open(filepath, 'w')
    file.write('{:<20}{:<15}{:<15}{:<15}{:<15}{:<15}\n'.format('Parameter', 'No. component', 'Optimized', 'Value', '-Error', '+Error'))
    for parameter_name in const['model_parameter_names']:
        parameter_indices = fitting_parameters['indices'][parameter_name]
        for i in range(len(parameter_indices)):
            file.write('{:<20}'.format(const['model_parameter_names_and_units'][parameter_name]))
            file.write('{:<15}'.format(i+1))
            parameter_object = parameter_indices[i]
            if parameter_object.optimize:
                file.write('{:<15}'.format('yes'))
            else:
                file.write('{:<15}'.format('no'))
            if parameter_object.optimize:
                parameter_value = optimized_model_parameters[parameter_object.index] / const['model_parameter_scales'][parameter_name]
                if parameter_name in const['angle_parameter_names']:
                    file.write('{:<15.1f}'.format(parameter_value))
                else:
                    file.write('{:<15.3f}'.format(parameter_value))
            else:
                parameter_value = fitting_parameters['values'][parameter_object.index]  / const['model_parameter_scales'][parameter_name]
                if parameter_name in const['angle_parameter_names']:
                    file.write('{:<15.1f}'.format(parameter_value))
                else:
                    file.write('{:<15.3f}'.format(parameter_value))
            if parameter_object.optimize:
                if model_parameter_errors != []:
                    if not np.isnan(model_parameter_errors[parameter_object.index][0]) and not np.isnan(model_parameter_errors[parameter_object.index][1]):
                        parameter_error = model_parameter_errors[parameter_object.index] / const['model_parameter_scales'][parameter_name]
                        if const['paired_model_parameters'][parameter_name] != 'none':
                            paired_parameter_name = const['paired_model_parameters'][parameter_name]
                            paired_parameter_object = fitting_parameters['indices'][paired_parameter_name][i]
                            if paired_parameter_object.optimize:
                                if np.isnan(model_parameter_errors[paired_parameter_object.index][0]) or np.isnan(model_parameter_errors[paired_parameter_object.index][1]):
                                    file.write('{:<15}{:<15}'.format('nan', 'nan'))
                                else:
                                    if parameter_name in const['angle_parameter_names']:
                                        file.write('{:<15.1f}{:<15.1f}'.format(parameter_error[0], parameter_error[1]))
                                    else:
                                        file.write('{:<15.3f}{:<15.3f}'.format(parameter_error[0], parameter_error[1]))
                            else:
                                if parameter_name in const['angle_parameter_names']:
                                    file.write('{:<15.1f}{:<15.1f}'.format(parameter_error[0], parameter_error[1]))
                                else:
                                    file.write('{:<15.3f}{:<15.3f}'.format(parameter_error[0], parameter_error[1]))
                        else:
                            if parameter_name in const['angle_parameter_names']:
                                file.write('{:<15.1f}{:<15.1f}'.format(parameter_error[0], parameter_error[1]))
                            else:
                                file.write('{:<15.3f}{:<15.3f}'.format(parameter_error[0], parameter_error[1]))
                    else:
                        file.write('{:<15}{:<15}'.format('nan', 'nan'))
                else:
                    file.write('{:<15}{:<15}'.format('nan', 'nan'))
            else:
                file.write('{:<15}{:<15}'.format('nan', 'nan')) 
            file.write('\n')
    file.close()


def save_model_parameters_multiple_runs(optimized_parameters_all_runs, fitting_parameters, filepath):    
    ''' Saves the multiple set of model parameters ''' 
    file = open(filepath, 'w')
    file.write('{:<20}{:<15}{:<15}'.format('Parameter', 'No. component', 'Optimized'))
    n_parameter_sets = len(optimized_parameters_all_runs)
    for i in range(n_parameter_sets):
        file.write('{:<15}'.format('Run ' + str(i+1)))
    file.write('\n')
    for parameter_name in const['model_parameter_names']:
        parameter_indices = fitting_parameters['indices'][parameter_name]
        for i in range(len(parameter_indices)):
            file.write('{:<20}'.format(const['model_parameter_names_and_units'][parameter_name]))
            file.write('{:<15}'.format(i+1))
            parameter_object = parameter_indices[i]
            if parameter_object.optimize:
                file.write('{:<15}'.format('yes'))
            else:
                file.write('{:<15}'.format('no'))
            if parameter_object.optimize:
                for k in range(n_parameter_sets):
                    parameter_value = optimized_parameters_all_runs[k][parameter_object.index] / const['model_parameter_scales'][parameter_name]
                    if parameter_name in const['angle_parameter_names']:
                        file.write('{:<15.1f}'.format(parameter_value))
                    else:
                        file.write('{:<15.3f}'.format(parameter_value))
            else:
                parameter_value = fitting_parameters['values'][parameter_object.index]  / const['model_parameter_scales'][parameter_name]
                for k in range(n_parameter_sets):
                    if parameter_name in const['angle_parameter_names']:
                        file.write('{:<15.1f}'.format(parameter_value))
                    else:
                        file.write('{:<15.3f}'.format(parameter_value))
            file.write('\n') 
    file.close()