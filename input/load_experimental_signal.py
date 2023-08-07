import numpy as np


def load_experimental_signal(filepath, column_numbers=[]):
    ''' Loads a PDS time trace from a file'''
    if column_numbers == []:
        column_numbers = [0, 1, 2]
    t = []
    s_re = []
    s_im = []
    file = open(filepath, 'r')
    for line in file:
        data_row = line.split()
        t.append(float(data_row[column_numbers[0]]))
        s_re.append(float(data_row[column_numbers[1]]))
        s_im.append(float(data_row[column_numbers[2]]))
    file.close()
    return np.array(t), np.array(s_re), np.array(s_im)