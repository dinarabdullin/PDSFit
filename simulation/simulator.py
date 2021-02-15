'''
The simulator class
'''

import numpy as np
from time import time

from spin_physics.spin import Spin
from mathematics.random_points_on_sphere import random_points_on_sphere
from mathematics.fibonacci_grid import fibonacci_grid_points
from mathematics.golden_spiral_grid import golden_spiral_grid_points
from mathematics.histogram import histogram

class Simulator():

    def __init__(self, calculation_settings):
        self.integration_method = calculation_settings["integration_method"]
        self.mc_sample_size = calculation_settings["mc_sample_size"]
        self.grid_size = calculation_settings["grid_size"]
        self.excitation_threshold = calculation_settings["excitation_treshold"]
        self.frequency_increment_epr_spectrum = 0.001 # in GHz
        self.field_directions = []
    
    def set_field_directions(self):
        field_directions = []
        if self.integration_method == "monte_carlo":
            field_directions = random_points_on_sphere(self.mc_sample_size)
        elif self.integration_method == "uniform_grids":
            ## Fibinacci grid points
            #field_directions = fibonacci_grid_points(self.grid_size["powder_averaging"])
            # Golden spiral grid points
            field_directions = golden_spiral_grid_points(self.grid_size["powder_averaging"])
        return field_directions
    
    def epr_spectrum(self, spin_system, field_value):
        # Random orientations of the static magnetic field
        a = time()
        if self.field_directions == []:
            self.field_directions = self.set_field_directions()
        num_field_directions = self.field_directions.shape[0]
        b = time()
        # Resonance frequencies and their probabilities
        all_frequencies = []
        all_probabilities = []
        for spin in spin_system:
            # Resonance frequencies
            resonance_frequencies = spin.res_freq(self.field_directions, field_value)
            weights = np.tile(spin.int_res_freq, (num_field_directions,1))
            # Frequency ranges
            min_resonance_frequency = np.amin(resonance_frequencies)
            max_resonance_frequency = np.amax(resonance_frequencies)
            # Spectrum
            frequencies = np.arange(np.around(min_resonance_frequency, 3), np.around(max_resonance_frequency)+self.frequency_increment_epr_spectrum, self.frequency_increment_epr_spectrum)
            probabilities = histogram(resonance_frequencies, bins=frequencies, weights=weights)
            all_frequencies.extend(frequencies)
            all_probabilities.extend(probabilities)
        c = time()
        # EPR spectrum
        min_frequency = np.amin(all_frequencies) - 0.150
        max_frequency = np.amax(all_frequencies) + 0.150
        spectrum = {}
        spectrum["f"] = np.arange(min_frequency, max_frequency+self.frequency_increment_epr_spectrum, self.frequency_increment_epr_spectrum)
        spectrum["s"] = histogram(all_frequencies, bins=spectrum["f"], weights=all_probabilities)
        d = time()
        print(b-a)
        print(c-b)
        print(d-c)
        return spectrum