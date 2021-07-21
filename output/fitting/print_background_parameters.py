import sys


def print_background_parameters(background_parameters, experiments, background):
    ''' Displays the background parameters '''
    sys.stdout.write('\nBackground parameters:\n')
    
    sys.stdout.write('{:<20}'.format('Experiment'))
    for parameter_name in background.parameter_names:
        sys.stdout.write('{:<20}'.format(background.parameter_full_names[parameter_name]))
    sys.stdout.write('\n')

    for i in range(len(experiments)):
        sys.stdout.write('{:<20}'.format(experiments[i].name))
        for parameter_name in background.parameter_names:
            sys.stdout.write('{:<20.6}'.format(background_parameters[i][parameter_name]))
        sys.stdout.write('\n')