''' Experiment class '''

from experiments.load_experimental_signal import load_experimental_signal

class Experiment:
    
    def __init__(self, name, technique, magnetic_field, detection_frequency, detection_pulse_lengths, pump_frequency, pump_pulse_lengths, mixing_time, temperature):
        self.name = name
        self.technique = technique
        self.magnetic_field = magnetic_field
        self.detection_frequency = detection_frequency
        self.detection_pulse_lengths = detection_pulse_lengths
        self.pump_frequency = pump_frequency
        self.pump_pulse_lengths = pump_pulse_lengths
        self.mixing_time = mixing_time
        self.temperature = temperature
        self.t = []
        self.s = []
        self.frequency_increment_bandwidth = 0.001 # in GHz
        
    def signal_from_file(self, filepath, signal_column):
        self.t, self.s = load_experimental_signal(filepath, signal_column)