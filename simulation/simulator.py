'''
The simulator class
'''

import numpy as np

from spin_physics.spin import Spin
from mathematics.random_points_on_sphere import random_points_on_sphere
from mathematics.fibonacci_grid import fibonacci_grid_points
from mathematics.golden_spiral_grid import golden_spiral_grid_points
from mathematics.histogram import histogram
from mathematics.values_from_distribution import values_from_distribution
from mathematics.spherical2cartesian import spherical2cartesian
from scipy.spatial.transform import Rotation 

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
        
    def set_field_orientations(self):
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
            self.field_orientations = self.set_field_orientations()
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
    def simulate_time_trace(self, experiment, spins, calculation_settings, parameters):
        time_trace = {}
        time_trace["t"] = experiment.t
        if len(spins) == 2:
            # Set the values of all geometric parameters and the J coupling constant
            distr = calculation_settings["distributions"]
            size = calculation_settings["mc_sample_size"]
            r = values_from_distribution(parameters['r_mean'][0][0], parameters['r_width'][0][0], distr['r'], size)
            xi = values_from_distribution(parameters['xi_mean'][0][0], parameters['xi_width'][0][0], distr['xi'], size)
            phi =values_from_distribution(parameters['phi_mean'][0][0], parameters['phi_width'][0][0], distr['phi'], size)
            alpha = values_from_distribution(parameters['alpha_mean'][0][0], parameters['alpha_width'][0][0], distr['alpha'], size)
            beta = values_from_distribution(parameters['beta_mean'][0][0], parameters['beta_width'][0][0], distr['beta'], size)
            gamma = values_from_distribution(parameters['gamma_mean'][0][0], parameters['gamma_width'][0][0], distr['gamma'], size)
            J = values_from_distribution(parameters['j_mean'][0][0], parameters['j_width'][0][0], distr['j'], size)
            fieldDirA = self.set_field_orientations()
            res_freqA = spins[0].res_freq(fieldDirA, experiment.magnetic_field)
            gValuesA = spins[0].g_effective(fieldDirA, size).reshape(size, )
            detProbA = experiment.detection_probability(res_freqA, spins[0].int_res_freq)
            pumpProbA = experiment.pump_probability(res_freqA,spins[0].int_res_freq)
            #Rotation matrix between the spin A and spin B frames
            rotationMatrices = Rotation.from_euler('ZXZ', np.column_stack((alpha, beta, gamma)))
            # Calculate the orientations of the magnetic field in the spin B frame
            fieldDirB = rotationMatrices.apply(fieldDirA)
            gValuesB = spins[1].g_effective(fieldDirB, size).reshape(size, )
            res_freqB = spins[1].res_freq(fieldDirB, experiment.magnetic_field)
            detProbB = experiment.detection_probability(res_freqB, spins[1].int_res_freq)
            # Calculate the probability of spin B to be excited by the pump pulse
            pumpProbB = experiment.pump_probability(res_freqB, spins[1].int_res_freq) * (detProbA > self.excitation_threshold)
            # Calculate the amplitude of the PELDOR signal
            amplitude = np.sum((detProbA > self.excitation_threshold) * detProbA + (detProbB > self.excitation_threshold) * detProbB)
            # Determine whether the spin pair is excited 
            excited_AB = (detProbA > self.excitation_threshold) * (pumpProbB > self.excitation_threshold)
            excited_BA = (detProbB > self.excitation_threshold) * (pumpProbA > self.excitation_threshold)
            excited_AB_and_BA = excited_AB * excited_BA
            excited_AB_and_notBA = excited_AB * np.logical_not(excited_BA)
            excited_BA_and_notAB = excited_BA * np.logical_not(excited_AB)
            # The direction of the distance vector
            distDirAB = spherical2cartesian(1, xi, phi)
            # The cosine of the dipolar angle (scalar product)
            cosDipolarAngle = np.sum(distDirAB*fieldDirA, axis = 1)
            # The dipolar frequency
            fdd = 52.04 * gValuesA * gValuesB * (1 - 3*cosDipolarAngle**2)/(2.0023**2 * r**3)
            # The echo modulation frequency //wdd in Paper 2014
            wdd = 2 * np.pi * (fdd + J)
            modAmplitude = excited_AB_and_BA * (detProbA * pumpProbB + detProbB * pumpProbA) 
            modAmplitude += excited_AB_and_notBA * (detProbA * pumpProbB) 
            modAmplitude += excited_BA_and_notAB * (detProbB * pumpProbA) 
            # The oscillating part of the PELDOR signal
            signalValues = np.zeros(time_trace["t"].size) 
            # for loop takes around 10 seconds. I've had problems trying to vectorize this because the array size would become very large 
            for i in range(time_trace["t"].size):
                signalValues[i] = np.sum(modAmplitude * (1-np.cos(wdd * time_trace["t"][i])))
            # Calculate the entire PELDOR signal and normalize it
            norm = 1/amplitude
            signalValues = norm * (amplitude-signalValues)
            time_trace['s'] = signalValues
        elif len(spins) == 3:
            pass
        else:
            raise ValueError('Invalid number of spins! Currently the number of spins is limited by 2 or 3.')
            sys.exit(1)
        return time_trace