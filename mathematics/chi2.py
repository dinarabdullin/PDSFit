import numpy as np


def chi2(x1, x2, sn=0):
    ''' Calculate chi2 between two signals '''
    chi2 = 0.0
    if sn:
        norm = 1 / sn**2
    else:
        norm = 1.0
    chi2 = norm * np.sum((x1 - x2)**2)  
    return chi2