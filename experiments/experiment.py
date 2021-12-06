import numpy as np
from input.load_experimental_signal import load_experimental_signal
from supplement.definitions import const
from mathematics.set_zero_point import set_zero_point
from mathematics.set_phase import set_phase


class Experiment:
    ''' Experiment class '''
    
    def __init__(self, name):
        self.name = name
        self.t = []
        self.s = []
        self.phase = 0.0
        self.zero_point = 0.0
        self.noise_std = 0.0
        
    def signal_from_file(self, filepath, column_numbers=[]):
        ''' Loads an experimental PDS time trace from a file'''
        t, s_re, s_im = load_experimental_signal(filepath, column_numbers)
        t = t - np.amin(t)
        t = const['ns2us'] * t
        phase, s_re, s_im = set_phase(s_re, s_im)
        zero_point, t, s_re, s_im = set_zero_point(t, s_re, s_im)
        noise_std = np.std(s_im)
        if noise_std < 1e-10:
            noise_std = 0
        self.phase = phase
        self.zero_point = zero_point
        self.t = t
        self.s = s_re
        self.s_im = s_im
        self.noise_std = noise_std
    
    def set_noise_std(self, noise_std):
        ''' Sets the standard deviation of noise in the experimental PDS time trace '''
        self.noise_std = noise_std