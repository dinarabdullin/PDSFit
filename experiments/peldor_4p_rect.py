'''
Class for 4-pulse ELDOR with rectangular pulses
'''

import numpy as np
from experiments.experiment import Experiment
from mathematics.values_from_distribution import values_from_distribution
from mathematics.spherical2cartesian import spherical2cartesian
from scipy.spatial.transform import Rotation 

class Peldor_4p_rect(Experiment):
    
    def __init__(self, name, technique, detection_frequency, detection_pulse_lengths, pump_frequency, pump_pulse_lengths, magnetic_field, temperature):
        super().__init__(name, technique, detection_frequency, detection_pulse_lengths, pump_frequency, pump_pulse_lengths, magnetic_field, temperature)
        self.detection_pi_half_pulse_length = self.detection_pulse_lengths[0]
        self.detection_pi_pulse_length = self.detection_pulse_lengths[1]
        self.pump_pulse_length = self.pump_pulse_lengths[0]
        self.bandwidth_detection_pi_half_pulse = 1 / (4 * self.detection_pi_half_pulse_length)
        self.bandwidth_detection_pi_pulse = 1 / (2 * self.detection_pi_pulse_length)
        self.bandwidth_pump_pulse = 1 / (2 * self.pump_pulse_length)
        
    def detection_probability(self, resonance_frequencies):
        frequency_offsets_squared = (self.detection_frequency - resonance_frequencies)**2
        rabi_frequencies_pi_half_pulse = np.sqrt(frequency_offsets_squared + self.bandwidth_detection_pi_half_pulse**2)
        rabi_frequencies_pi_pulse = np.sqrt(frequency_offsets_squared + self.bandwidth_detection_pi_pulse**2)
        detection_probabilities = (self.bandwidth_detection_pi_half_pulse / rabi_frequencies_pi_half_pulse) * np.sin(2*np.pi * rabi_frequencies_pi_half_pulse * self.detection_pi_half_pulse_length) * \
                                  0.25 * (self.bandwidth_detection_pi_pulse / rabi_frequencies_pi_pulse) ** 4 * (1 - np.cos(2*np.pi * rabi_frequencies_pi_pulse * self.detection_pi_pulse_length))**2
        return detection_probabilities
    
    def pump_probability(self, resonance_frequencies):
        frequency_offsets_squared = (self.pump_frequency - resonance_frequencies)**2
        rabi_frequencies_pump_pulse = np.sqrt(frequency_offsets_squared + self.bandwidth_pump_pulse**2)
        pump_probabilities = 0.5 * (self.bandwidth_pump_pulse / rabi_frequencies_pump_pulse)**2 * (1 - np.cos(2*np.pi * rabi_frequencies_pump_pulse * self.pump_pulse_length))
        return pump_probabilities
        
    def get_detection_bandwidth(self, ranges=()):
        if ranges == ():
            frequencies = np.arange(self.detection_frequency - 10 * self.bandwidth_detection_pi_half_pulse, self.detection_frequency + 10 * self.bandwidth_detection_pi_half_pulse, self.frequency_increment_bandwidth)
        else:
            frequencies = np.arange(ranges[0], ranges[1], self.frequency_increment_bandwidth)
        probabilities = self.detection_probability(frequencies)
        detection_bandwidth = {}
        detection_bandwidth["f"] = frequencies
        detection_bandwidth["p"] = probabilities
        return detection_bandwidth
    
    def get_pump_bandwidth(self, ranges=()):
        if ranges == ():
            frequencies = np.arange(self.pump_frequency - 10 * self.bandwidth_pump_pulse, self.pump_frequency + 10 * self.bandwidth_pump_pulse, self.frequency_increment_bandwidth)
        else:
            frequencies = np.arange(ranges[0], ranges[1], self.frequency_increment_bandwidth)
        probabilities = self.pump_probability(frequencies)
        pump_bandwidth = {}
        pump_bandwidth["f"] = frequencies
        pump_bandwidth["p"] = probabilities
        return pump_bandwidth
    
    
    
    # TODO This function should calculate a 4pELDOR time trace using dipolar frequencies, detection probabilities, pump probabilities, etc
    # pumpEfficiency needs to be added to cfg file?
    def compute_time_trace(self, simulator, spins, parameters, calculation_settings):
        # replace with scipy soon
        # Calculate the PELDOR signal for the two-spin sytem
        if len(spins) == 2:
            # Set the values of all geometric parameters and the J coupling constant
            distr = calculation_settings["distributions"]
            size = calculation_settings["mc_sample_size"]
            r = values_from_distribution(parameters['r_mean'], parameters['r_width'], distr['r'], size)
            xi = values_from_distribution(parameters['xi_mean'], parameters['xi_width'], distr['xi'], size)
            phi =values_from_distribution(parameters['phi_mean'], parameters['phi_width'], distr['phi'], size)
            alpha = values_from_distribution(parameters['alpha_mean'], parameters['alpha_width'], distr['alpha'], size)
            beta = values_from_distribution(parameters['beta_mean'], parameters['beta_width'], distr['beta'], size)
            gamma = values_from_distribution(parameters['gamma_mean'], parameters['gamma_width'], distr['gamma'], size)
            J = values_from_distribution(parameters['j_mean'], parameters['j_width'], distr['j'], size)

            fieldDirA = simulator.set_field_directions()
            res_freqA = spins[0].res_freq(fieldDirA, self.magnetic_field)
            gValuesA = spins[0].g_effective(fieldDirA, size)
            detProbA = self.detection_probability(res_freqA)
            pumpProbA = self.pump_probability(res_freqA)

            #Rotation matrix between the spin A and spin B frames
            rotationMatrix = Rotation.from_euler('ZXZ', [alpha, beta, gamma], degrees = False)
            # Calculate the directions of the magnetic field in the spin B frame
            fieldDirB = rotationMatrix.apply(fieldDirA)
            gValuesB = spins[1].g_effective(fieldDirB, size)
            res_freqB = spins[1].res_freq(fieldDirB, self.magnetic_field)
            detProbB = self.detection_probability(res_freqB)
            # Calculate the probability of spin B to be excited by the pump pulse
            pumpProbB = self.pump_probability(res_freqB) * (detProbA > simulator.excitation_threshold)
            # Calculate the amplitude of the PELDOR signal
            amplitude = np.sum((detProbA > simulator.excitation_threshold) * detProbA + (detProbB > simulator.excitation_threshold) * detProbB)
            # Determine whether the spin pair is excited 
            excited_AB = (detProbA > simulator.excitation_threshold) * (pumpProbB > simulator.excitation_threshold)
            excited_BA = (detProbB > simulator.excitation_threshold) * (pumpProbA > simulator.excitation_threshold)
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
            ############
            pumpEfficiency = 0.9   # where can this value be taken from? cfg file?
            ############
            modAmplitude = excited_AB_and_BA * (detProbA * pumpProbB + detProbB * pumpProbA) * pumpEfficiency
            modAmplitude += excited_AB_and_notBA * (detProbA * pumpProbB) * pumpEfficiency
            modAmplitude += excited_BA_and_notAB * (detProbB * pumpProbA) * pumpEfficiency
            # The oscillating part of the PELDOR signal
            signalValues = np.zeros(self.t.size)  #maybe we should rename t to something more meaningful like timeValues 
            for i in self.t.size:
                signalValues[i] = np.sum(modAmplitude * (1-np.cos(wdd * self.t[i])))
            # Calculate the entire PELDOR signal and normalize it
            norm = 1/amplitude
            signalValues = norm * (amplitude-signalValues)
        # Implement later
        if len(spins) == 3:
            pass
        return signalValues
