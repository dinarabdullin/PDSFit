import sys


def print_background_parameters(background_parameters, experiments):
    ''' Displays the background parameters '''
    sys.stdout.write('\nBackground parameters:\n')
    sys.stdout.write('{:<20}{:<20}{:<20}{:<20}\n'.format('Experiment', 'Decay constant', 'Dimension', 'Scale factor'))
    for i in range(len(experiments)):
        sys.stdout.write('{:<20}'.format(experiments[i].name))
        sys.stdout.write('{:<20.6}'.format(background_parameters[i]['decay_constant']))
        sys.stdout.write('{:<20.2}'.format(background_parameters[i]['dimension']))
        sys.stdout.write('{:<20.3}'.format(background_parameters[i]['scale_factor']))
        sys.stdout.write('\n')