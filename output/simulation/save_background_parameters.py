def save_background_parameters(background_parameters, background, experiments, filepath):
    ''' Saves optimized background parameters '''
    file = open(filepath, 'w')
    file.write('{:<20}'.format('Experiment'))
    for parameter_name in background.parameter_names:
        file.write('{:<20}'.format(background.parameter_full_names[parameter_name]))
    file.write('\n')
    for i in range(len(experiments)):
        file.write('{:<20}'.format(experiments[i].name))
        for parameter_name in background.parameter_names:
            file.write('{:<20.6}'.format(background_parameters[i][parameter_name]))
        file.write('\n')
    file.close()