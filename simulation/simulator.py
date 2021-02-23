'''
The simulator class
'''

import numpy as np

from spin_physics.spin import Spin
from mathematics.random_points_on_sphere import random_points_on_sphere
from mathematics.fibonacci_grid import fibonacci_grid_points
from mathematics.golden_spiral_grid import golden_spiral_grid_points
from mathematics.histogram import histogram

class Simulator():

    def __init__(self, calculation_settings):
        self.integration_method = calculation_settings["integration_method"]
        if self.integration_method == "monte_carlo":
            self.separate_grids = False
        else:
            self.separate_grids = True
        self.mc_sample_size = calculation_settings["mc_sample_size"]
        self.grid_size = calculation_settings["grid_size"]
        self.distributions = calculation_settings["distributions"]
        self.excitation_threshold = calculation_settings["excitation_treshold"]
        self.frequency_increment_epr_spectrum = 0.001 # in GHz
        self.field_orientations = []
        self.euler_angles_convention = "ZXZ" 
        
    def field_orientations(self):
        field_orientations = []
        if self.integration_method == "monte_carlo":
            field_orientations = random_points_on_sphere(self.mc_sample_size)
        elif self.integration_method == "uniform_grids":
            ## Fibinacci grid points
            #field_orientations = fibonacci_grid_points(self.grid_size["powder_averaging"])
            # Golden spiral grid points
            field_orientations = golden_spiral_grid_points(self.grid_size["powder_averaging"])
        return field_orientations

    def epr_spectrum(self, spins, field_value):
        # Random orientations of the static magnetic field
        if self.field_orientations == []:
            self.field_orientations = self.field_orientations()
        num_field_orientations = self.field_orientations.shape[0]
        # Resonance frequencies and their probabilities
        all_frequencies = []
        all_probabilities = []
        for spin in spins:
            # Resonance frequencies
            resonance_frequencies = spin.res_freq(self.field_orientations, field_value)
            weights = np.tile(spin.int_res_freq, (num_field_orientations,1))
            # Frequency ranges
            min_resonance_frequency = np.amin(resonance_frequencies)
            max_resonance_frequency = np.amax(resonance_frequencies)
            # Spectrum
            frequencies = np.arange(np.around(min_resonance_frequency, 3), np.around(max_resonance_frequency)+self.frequency_increment_epr_spectrum, self.frequency_increment_epr_spectrum)
            probabilities = histogram(resonance_frequencies, bins=frequencies, weights=weights)
            all_frequencies.extend(frequencies)
            all_probabilities.extend(probabilities)
        # EPR spectrum
        min_frequency = np.amin(all_frequencies) - 0.150
        max_frequency = np.amax(all_frequencies) + 0.150
        spectrum = {}
        spectrum["f"] = np.arange(min_frequency, max_frequency+self.frequency_increment_epr_spectrum, self.frequency_increment_epr_spectrum)
        spectrum["s"] = histogram(all_frequencies, bins=spectrum["f"], weights=all_probabilities)
        return spectrum
    
    # IN PROGRESS
    def simulate_time_trace(self, experiment, spins, variables):
        time_trace = {}
        time_trace["t"] = experiment.t
        if len(spins) == 2:
            pass # put your simulation code here instead of pass
        elif len(spins) == 3:
            pass
        else:
            raise ValueError('Invalid number of spins! Currently the number of spins is limited by 2 or 3.')
            sys.exit(1)
        return time_trace