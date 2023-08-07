def save_epr_spectrum(spectrum, filepath):
    ''' Saves the simulated EPR spectrum of a spin system'''
    file = open(filepath, 'w')
    for i in range(spectrum['f'].size):
        file.write('{0:<20.3f} {1:<20.6f} \n'.format(spectrum['f'][i], spectrum['p'][i]))
    file.close()