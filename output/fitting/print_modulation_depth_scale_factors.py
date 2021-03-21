import sys


def print_modulation_depth_scale_factors(modulation_depth_scale_factors, experiments):
    ''' Displays the scale factors of modulation depths '''
    sys.stdout.write('\nScale factors of modulation depths:\n')
    sys.stdout.write("{:<20}{:<15}\n".format('Experiment', 'Scale factor'))
    for i in range(len(experiments)):
        sys.stdout.write('{:<20}'.format(experiments[i].name))
        sys.stdout.write('{:<15.4}'.format(modulation_depth_scale_factors[i]))
        sys.stdout.write('\n')