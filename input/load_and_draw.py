'''
Description is missing
'''

numpy as np
import matplotlib.pyplot as plt

filepath = "/home/pablo/Documents/PeldorFit/osPDSFit/source_code/tests/results/06-Feb-2021--20:55/spectrum.dat"

def load_spectrum(filepath):
    x_list = []
    y_list = []
    with open(filepath, 'r') as f:
        next(f)
        for line in f:
            str = line.split()
            x_list.append(float(str[0]))
            y_list.append(float(str[1]))
    x_array = np.array(x_list)	
    y_array = np.array(y_list)
    return [x_array, y_array]

def draw_spectrum(filepath):
    spectrum = load_spectrum(filepath)
    x = spectrum[0]
    y = spectrum[1]
    plt.plot(x, y)
    plt.show()

draw_spectrum(filepath)