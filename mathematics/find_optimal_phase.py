import numpy as np
from supplement.definitions import const


def find_optimal_phase(y_re, y_im):
    ''' Finds the optimal phase of a PDS time trace '''
    # Make a linear sphase grid and compute cosines and sines
    phase_increment = 0.1 
    phases = np.arange(-90, 90+phase_increment, phase_increment)
    cosines = np.cos(phases * const['deg2rad'])
    sines = np.sin(phases * const['deg2rad'])
    # Apply the phase correction to the imaginary part of the PDS time trace
    n_phases = phases.size
    n_datapoints = int(2/3 * float(y_re.size))
    index_start = y_re.size - n_datapoints - 1
    y_im_phased = cosines.reshape(n_phases, 1) * y_im[index_start:-1].reshape(1, n_datapoints) + \
                  sines.reshape(n_phases, 1) * y_re[index_start:-1].reshape(1, n_datapoints)
    # Minimize the mean of the imaginary part of the PDS time trace  
    y_im_mean = np.abs(np.mean(y_im_phased, axis=1))
    index_min = np.argmin(y_im_mean)
    if type(index_min) is np.ndarray:
        index_min = index_min[0]    
    # Optimal phase
    phase = phases[index_min]
    return phase
    

def set_phase(y_re, y_im, phase):    
    ''' Sets the phase of a PDS time trace '''
    cosine = np.cos(phase * const['deg2rad'])
    sine = np.sin(phase * const['deg2rad'])
    y_re_new = y_re * cosine - y_im * sine
    y_im_new = y_im * cosine + y_re * sine
    # from plots.preprocessing.plot_phase import plot_phase
    # plot_phase(phases, y_im_mean, y_re, y_im, y_re_new, y_im_new)
    return y_re_new, y_im_new