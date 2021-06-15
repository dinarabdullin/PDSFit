import sys
import numpy as np


def compare_size(list1, list2, name1, name2):
    ''' Compares the size (shape) of two lists '''
    array1 = np.array(list1)
    array2 = np.array(list2)
    if array1.shape != array2.shape:
        raise ValueError('Parameters \'{0}\' and \'{1}\' must have same dimensions!'.format(name1, name2))
        sys.exit(1)