''' Load an experimental signal '''

import numpy as np

def load_experimental_signal(filepath, signal_column):
    x = []
    y = []
    file = open(filepath, 'r')
    for line in file:
        str = line.split()
        x.append( float(str[0]) )
        y.append( float(str[signal_column]) )
    file.close()
    return np.array(x), np.array(y)
