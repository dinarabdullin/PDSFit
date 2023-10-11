import sys
import numpy as np
from preprocessing.preprocessing import find_optimal_phase, set_phase, \
    find_zero_point, set_zero_point, compute_noise_level
from supplement.definitions import const


class Experiment:
    """PDS experiment."""
    
    def __init__(self, name):
        self.name = name
    
    
    def load_signal_from_file(self, filepath, column_numbers = [0, 1, 2]):
        """Load a PDS time trace from a file."""
        t, s_re, s_im = [], [], []
        file = open(filepath, "r")
        for line in file:
            data_row = line.split()
            t.append(float(data_row[column_numbers[0]]))
            s_re.append(float(data_row[column_numbers[1]]))
            s_im.append(float(data_row[column_numbers[2]]))
        file.close()
        self.t = np.array(t)
        self.s = np.array(s_re)
        self.s_im = np.array(s_im)
    
    
    def perform_preprocessing(
        self, phase = np.nan, zero_point = np.nan, noise_std = np.nan
        ):
        """Preprocess a PDS time trace."""
        t, s_re, s_im = self.t, self.s, self.s_im
        # Set the first time point to 0
        t = t - np.amin(t)
        # Set the units of the time axis to microseconds
        t = const["ns2us"] * t
        # Find the optimal phase
        if not np.isnan(phase):
            ph = phase
        else:
            ph = find_optimal_phase(s_re, s_im)
        s_re, s_im = set_phase(s_re, s_im, ph)
        # Normalize to 1
        s_re_max = np.amax(s_re) 
        s_re = s_re / s_re_max
        s_im = s_im / s_re_max
        # Find / set the zero point and re-normalize to 1
        if not np.isnan(zero_point):
            t_zp = const["ns2us"] * zero_point
        else:
            t_zp = find_zero_point(t, s_re)
        t, s_re, s_im = set_zero_point(t, s_re, s_im, t_zp)
        # Compute / set the noise level 
        if not np.isnan(noise_std):
            noise_level = noise_std
        else:
            noise_level = compute_noise_level(s_im)
        if noise_level == 0:
            raise ValueError(
                "Error: The zero level of noise is encountered!\n\
                Specify the nonzero quadrature component of the PDS time trace or\n\
                provide the noise level explicitly via noise_std."
                )
            sys.exit(1)
        # Store the data
        self.t = t
        self.s = s_re
        self.s_im = s_im
        self.phase = ph
        self.zero_point = t_zp
        self.noise_std = noise_level