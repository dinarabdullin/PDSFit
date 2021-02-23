'''
Class for 4-pulse ELDOR with rectangular pulses
'''

from time import time
import numpy as np
from experiments.experiment import Experiment

class Ridme_5p_rect(Experiment):
    
    def __init__(self, name, technique, detection_frequency, detection_pulse_lengths, pump_frequency, pump_pulse_lengths, magnetic_field, temperature):
        super().__init__(name, technique, detection_frequency, detection_pulse_lengths, pump_frequency, pump_pulse_lengths, magnetic_field, temperature)
        self.detection_pi_half_pulse_length = self.detection_pulse_lengths[0]
        self.detection_pi_pulse_length = self.detection_pulse_lengths[1]
        self.bandwidth_detection_pi_half_pulse = 1 / (4 * self.detection_pi_half_pulse_length)
        self.bandwidth_detection_pi_pulse = 1 / (2 * self.detection_pi_pulse_length)
        self.double_frequency_experiment = False
        
    def detection_probability(self, resonance_frequencies, weights):
        frequency_offsets_squared = (self.detection_frequency - resonance_frequencies)**2
        rabi_frequencies_pi_half_pulse = np.sqrt(frequency_offsets_squared + self.bandwidth_detection_pi_half_pulse**2)
        rabi_frequencies_pi_pulse = np.sqrt(frequency_offsets_squared + self.bandwidth_detection_pi_pulse**2)
        # This equation needs to be validated by theory!!!
        detection_probabilities = (self.bandwidth_detection_pi_half_pulse / rabi_frequencies_pi_half_pulse)**3 * np.sin(2*np.pi * rabi_frequencies_pi_half_pulse * self.detection_pi_half_pulse_length)**3 * \
                                  0.25 * (self.bandwidth_detection_pi_pulse / rabi_frequencies_pi_pulse) ** 4 * (1 - np.cos(2*np.pi * rabi_frequencies_pi_pulse * self.detection_pi_pulse_length))**2
        detection_probabilities = detection_probabilities * weights.reshape(weights.size,1)
        detection_probabilities = detection_probabilities.sum(axis=0)
        return detection_probabilities

    def pump_probability(self):
        return 0.5
        
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