import numpy as np


def save_background_parameters(background_parameters, background_parameter_errors, background, experiments, filepath):
    ''' Saves optimized background parameters and their errors '''
    file = open(filepath, 'w')
    file.write('{:<20}{:<15}{:<15}{:<15}{:<15}{:<15}\n'.format('Parameter', 'Experiment', 'Optimized', 'Value', '-Error', '+Error'))
    for parameter_name in background.parameter_names:
        for i in range(len(experiments)):
            file.write('{:<20}'.format(background.parameter_full_names[parameter_name]))
            file.write('{:<15}'.format(experiments[i].name))
            if background.parameters[parameter_name]['optimize']:
                file.write('{:<15}'.format('yes'))
                background_parameter_value = background_parameters[i][parameter_name]
                if background_parameter_errors != []:
                    background_parameter_error = background_parameter_errors[i][parameter_name]
                else:
                    background_parameter_error = np.array([np.nan, np.nan])
            else:
                file.write('{:<15}'.format('no'))
                background_parameter_value = background.parameters[parameter_name]['value']
                background_parameter_error = np.array([np.nan, np.nan])
            file.write('{:<15.6f}'.format(background_parameter_value)) 
            if not np.isnan(background_parameter_error[0]) and not np.isnan(background_parameter_error[1]):
                file.write('{:<15.6f}{:<15.6f}'.format(background_parameter_error[0], background_parameter_error[1])) 
            else:
                file.write('{:<15}{:<15}'.format('nan', 'nan'))
            file.write('\n')
    file.close()