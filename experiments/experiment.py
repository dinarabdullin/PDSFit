import numpy as np
from experiments.load_experimental_signal import load_experimental_signal


class Experiment:
    ''' Experiment class '''
    
    def __init__(self, name):
        self.name = name
        self.t = []
        self.s = []
        self.modulation_depth = 0.0
        self.frequency_increment_bandwidth = 0.001 # in GHz
        
    def signal_from_file(self, filepath, signal_column):
        ''' Load an experimental PDS time trace from a file'''
        self.t, self.s = load_experimental_signal(filepath, signal_column)
        if self.t[0] != 0:
            self.t = self.t - self.t[0]
    
    def set_noise_std(self, noise_std):
        ''' Set the standard deviation of noise in the experimental PDS time trace '''
        self.noise_std = noise_std
    
    def compute_modulation_depth(self, interval):
        ''' Computes the modulation depth of a PDS time trace '''
        length_t_axis = self.t[-1] - self.t[0]
        if interval > length_t_axis:
            raise ValueError('Invalid interval for calculation of modulation depth!')
            sys.exit(1)
        else:
            t_start = self.t[-1] - interval
            idx_start = (np.abs(self.t - t_start)).argmin()
            self.modulation_depth = 1.0 - np.mean(self.s[idx_start:-1])