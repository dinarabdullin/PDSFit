import numpy as np
from input.load_experimental_signal import load_experimental_signal
from supplement.definitions import const
from mathematics.find_optimal_phase import find_optimal_phase
from mathematics.find_zero_point import find_zero_point, set_zero_point
from mathematics.compute_noise_level import compute_noise_level


class Experiment:
    ''' Experiment class '''
    
    def __init__(self, name):
        self.name = name
        self.t = []
        self.s = []
        self.s_im = []
        self.phase = 0.0
        self.zero_point = 0.0
        self.noise_std = 0.0
        
    def signal_from_file(self, filepath, zero_point=np.nan, noise_std=np.nan, column_numbers=[]):
        ''' Loads an experimental PDS time trace from a file'''
        # Load the PDS time trace
        t, s_re, s_im = load_experimental_signal(filepath, column_numbers)
        # Set the first time point to 0
        t = t - np.amin(t)
        # Set the units of the time axis to microseconds
        t = const['ns2us'] * t
        # Find the optimal phase and normalize to 1
        ph, s_re, s_im = find_optimal_phase(s_re, s_im)
        s_re_max = np.amax(s_re) 
        s_re = s_re / s_re_max
        s_im = s_im / s_re_max
        # Find / set the zero point and re-normalize to 1
        if not np.isnan(zero_point):
            t_zp = const['ns2us'] * zero_point
        else:
            t_zp = find_zero_point(t, s_re)
        t, s_re, s_im = set_zero_point(t, s_re, s_im, t_zp)
        # Compute / set the noise level 
        if not np.isnan(noise_std):
            nl = noise_std
        else:
            nl = compute_noise_level(s_im)
        if nl < 1e-10:
            nl = 0
        # Store the data
        self.t = t
        self.s = s_re
        self.s_im = s_im
        self.phase = ph
        self.zero_point = t_zp
        self.noise_std = nl
    
    def set_noise_std(self, noise_std):
        ''' Sets the standard deviation of noise in the experimental PDS time trace '''
        self.noise_std = noise_std