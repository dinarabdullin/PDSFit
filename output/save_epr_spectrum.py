'''
Save a simulated EPR spectrum
'''

def save_epr_spectrum(spectrum, directory):
    filepath = directory + "epr_spectrum.dat"
    file = open(filepath, 'w')
    for i in range(spectrum["f"].size):
        file.write('{0:<12.4f} {1:<12.4f} \n'.format(spectrum["f"][i], spectrum["s"][i]))
    file.close()