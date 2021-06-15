import numpy as np
from supplement.definitions import const


def set_phase(y_re, y_im):
    ''' Sets the phase of the PDS time trace '''
    # Make a linear sphase grid and compute cosines and sines
    phase_increment = 0.1 
    phases = np.arange(-180, 180+phase_increment, phase_increment)
    cosines = np.cos(phases * const['deg2rad'])
    sines = np.sin(phases * const['deg2rad'])
    # Apply the phase correction to the imaginary part of the PDS time trace
    n_phases = phases.size
    n_datapoints = int(0.75 * float(y_re.size))
    index_start = y_re.size - n_datapoints - 1
    y_im_phased = cosines.reshape(n_phases, 1) * y_im[index_start:-1].reshape(1, n_datapoints) + sines.reshape(n_phases, 1) * y_re[index_start:-1].reshape(1, n_datapoints)
    # Minimize the std of the imaginary part of the PDS time trace
    y_im_std = np.std(y_im_phased, axis=1)
    index_min = np.argmin(y_im_std)
    if type(index_min) is np.ndarray:
        index_min = index_min[0]
    # Set the optimal phase
    phase = phases[index_min]
    cosine = np.cos(phase * const['deg2rad'])
    sine = np.sin(phase * const['deg2rad'])
    y_re_new = y_re * cosine - y_im * sine
    y_im_new = y_im * cosine + y_re * sine
    y_re_max = np.amax(y_re_new) 
    y_re_new = y_re_new / y_re_max
    y_im_new = y_im_new / y_re_max
    return phase, y_re_new, y_im_new