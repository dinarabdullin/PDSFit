def save_epr_spectrum(spectrum, filepath):
    ''' Saves a simulated EPR spectrum '''
    file = open(filepath, 'w')
    for i in range(spectrum['f'].size):
        file.write('{0:<15.4f} {1:<15.4f} \n'.format(spectrum['f'][i], spectrum['p'][i]))
    file.close()