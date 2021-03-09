def save_epr_spectrum(spectrum, directory, experiment_name):
    ''' Save a simulated EPR spectrum '''
    filepath = directory + 'epr_spectrum_' + experiment_name + '.dat'
    file = open(filepath, 'w')
    for i in range(spectrum['f'].size):
        file.write('{0:<12.4f} {1:<12.4f} \n'.format(spectrum['f'][i], spectrum['p'][i]))
    file.close()