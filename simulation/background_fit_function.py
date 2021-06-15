import numpy as np


def background_fit_function(t, k, d, s, V):
    ''' A function that describes dependence of a PDS time trace on background parameters '''
    return np.exp(-k * np.abs(t)**(d/3)) * (np.ones(V.size) + s * (V - np.ones(V.size)))

def background_fit_function_kds_wrapper(t, k, d, s, V):
    return background_fit_function(t, k, d, s, V)
    
def background_fit_function_ksd_wrapper(t, k, s, d, V):
    return background_fit_function(t, k, d, s, V)

def background_fit_function_skd_wrapper(t, s, k, d, V):
    return background_fit_function(t, k, d, s, V)
    
def background_fit_function_dsk_wrapper(t, d, s, k, V):
    return background_fit_function(t, k, d, s, V)

def background_fit_function_dks_wrapper(t, d, k, s, V):
    return background_fit_function(t, k, d, s, V)