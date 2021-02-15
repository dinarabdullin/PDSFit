'''
Class for 4-pulse ELDOR with rectangular pulses
'''

import numpy as np
from experiments.experiment import Experiment

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
    def compute_time_trace(self):
        pass
