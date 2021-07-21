from time import time
import numpy as np
from experiments.experiment import Experiment


class Peldor_4p_rect(Experiment):
    ''' Class for 4-pulse ELDOR with rectangular pulses '''
    
    def __init__(self, name):
        super().__init__(name)
        self.technique = 'peldor'
        self.parameter_names = {
            'magnetic_field': 'float', 
            'detection_frequency': 'float', 
            'detection_pulse_lengths': 'float_array', 
            'pump_frequency': 'float', 
            'pump_pulse_lengths': 'float_array'
            }
        self.frequency_increment_bandwidth = 0.001 # in GHz
    
    def set_parameters(self, parameter_values):
        ''' Sets the parameters of an experiment '''
        self.magnetic_field = parameter_values['magnetic_field'] 
        self.detection_frequency = parameter_values['detection_frequency'] 
        self.detection_pi_half_pulse_length = parameter_values['detection_pulse_lengths'][0]
        self.detection_pi_pulse_length = parameter_values['detection_pulse_lengths'][1]
        self.pump_frequency = parameter_values['pump_frequency'] 
        self.pump_pulse_length = parameter_values['pump_pulse_lengths'][0]
        self.bandwidth_detection_pi_half_pulse = 1 / (4 * self.detection_pi_half_pulse_length)
        self.bandwidth_detection_pi_pulse = 1 / (2 * self.detection_pi_pulse_length)
        self.bandwidth_pump_pulse = 1 / (2 * self.pump_pulse_length)
    
    def detection_probability(self, resonance_frequencies, weights=[]):
        ''' Computes detection probabilities for different resonance frequencies '''
        frequency_offsets_squared = (self.detection_frequency - resonance_frequencies)**2
        rabi_frequencies_pi_half_pulse = np.sqrt(frequency_offsets_squared + self.bandwidth_detection_pi_half_pulse**2)
        rabi_frequencies_pi_pulse = np.sqrt(frequency_offsets_squared + self.bandwidth_detection_pi_pulse**2)
        detection_probabilities = (self.bandwidth_detection_pi_half_pulse / rabi_frequencies_pi_half_pulse) * np.sin(2*np.pi * rabi_frequencies_pi_half_pulse * self.detection_pi_half_pulse_length) * \
                                  0.25 * (self.bandwidth_detection_pi_pulse / rabi_frequencies_pi_pulse) ** 4 * (1 - np.cos(2*np.pi * rabi_frequencies_pi_pulse * self.detection_pi_pulse_length))**2
        if weights != []:
            if isinstance(weights, list):
                weights = np.array(weights).reshape(len(weights),1)
            detection_probabilities = detection_probabilities * weights
            detection_probabilities = detection_probabilities.sum(axis=1)
        return detection_probabilities.flatten()

    def pump_probability(self, resonance_frequencies, weights=[]):
        ''' Computes pump probabilities for different resonance frequencies '''
        frequency_offsets_squared = (self.pump_frequency - resonance_frequencies)**2
        rabi_frequencies_pump_pulse = np.sqrt(frequency_offsets_squared + self.bandwidth_pump_pulse**2)
        pump_probabilities = 0.5 * (self.bandwidth_pump_pulse / rabi_frequencies_pump_pulse)**2 * (1 - np.cos(2*np.pi * rabi_frequencies_pump_pulse * self.pump_pulse_length))
        if weights != []:
            if isinstance(weights, list):
                weights = np.array(weights).reshape(len(weights),1)
            pump_probabilities = pump_probabilities * weights
            pump_probabilities = pump_probabilities.sum(axis=1)
        return pump_probabilities.flatten()
        
    def get_detection_bandwidth(self, ranges=()):
        ''' Computes the bandwidth of detection pulses '''
        if ranges == ():
            frequencies = np.arange(self.detection_frequency - 10 * self.bandwidth_detection_pi_half_pulse, self.detection_frequency + 10 * self.bandwidth_detection_pi_half_pulse, self.frequency_increment_bandwidth)
        else:
            frequencies = np.arange(ranges[0], ranges[1], self.frequency_increment_bandwidth)
        probabilities = self.detection_probability(frequencies)
        detection_bandwidth = {}
        detection_bandwidth['f'] = frequencies
        detection_bandwidth['p'] = probabilities
        return detection_bandwidth

    def get_pump_bandwidth(self, ranges=()):
        ''' Computes the bandwidth of a pump pulse '''
        if ranges == ():
            frequencies = np.arange(self.pump_frequency - 10 * self.bandwidth_pump_pulse, self.pump_frequency + 10 * self.bandwidth_pump_pulse, self.frequency_increment_bandwidth)
        else:
            frequencies = np.arange(ranges[0], ranges[1], self.frequency_increment_bandwidth)
        probabilities = self.pump_probability(frequencies)
        pump_bandwidth = {}
        pump_bandwidth['f'] = frequencies
        pump_bandwidth['p'] = probabilities
        return pump_bandwidth