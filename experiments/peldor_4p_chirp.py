import numpy as np
from experiments.experiment import Experiment
from mathematics.find_nearest import find_nearest


class Peldor_4p_chirp(Experiment):
    """Four-pulse ELDOR experiment with rectangular detection pulses and a chirp pump pulse."""
    
    def __init__(self, name):
        super().__init__(name)
        self.technique = "peldor"
        self.parameter_names = {
            "magnetic_field": "float",
            "detection_frequency": "float",
            "detection_pulse_lengths": "float", 
            "pump_frequency": "float",
            "pump_frequency_width": "float",
            "pump_pulse_lengths": "float", 
            "pump_pulse_rise_times": "float",
            "critical_adiabaticity": "float"
            }
        self.frequency_increment = 0.001 # in GHz
        self.time_increment = 0.1 # in s
    
    
    def set_parameters(self, parameter_values):
        """Set the parameters of an experiment."""
        self.magnetic_field = parameter_values["magnetic_field"]
        self.detection_frequency = parameter_values["detection_frequency"]
        self.detection_pi_half_pulse_length = parameter_values["detection_pulse_lengths"][0]
        self.detection_pi_pulse_length = parameter_values["detection_pulse_lengths"][1]
        self.pump_frequency = parameter_values["pump_frequency"]
        self.pump_frequency_width = parameter_values["pump_frequency_width"]
        self.pump_pulse_length = parameter_values["pump_pulse_lengths"][0]
        self.pump_pulse_rise_time = parameter_values["pump_pulse_rise_times"][0]
        self.critical_adiabaticity = parameter_values["critical_adiabaticity"]
        self.detection_bandwidth = self.compute_detection_bandwidth()
        self.pump_bandwidth = self.compute_pump_bandwidth()
    
    
    def compute_detection_bandwidth(self):
        """Compute the bandwidth of detection pulses."""
        bandwidth_detection_pi_half_pulse = 1 / (4 * self.detection_pi_half_pulse_length)
        bandwidth_detection_pi_pulse = 1 / (2 * self.detection_pi_pulse_length)  
        frequency_axis = np.arange(
            self.detection_frequency - 10 * bandwidth_detection_pi_half_pulse, 
            self.detection_frequency + 10 * bandwidth_detection_pi_half_pulse, 
            self.frequency_increment
            )
        frequency_offsets_squared = (self.detection_frequency - frequency_axis)**2
        rabi_frequencies_pi_half_pulse = np.sqrt(frequency_offsets_squared + bandwidth_detection_pi_half_pulse**2)
        rabi_frequencies_pi_pulse = np.sqrt(frequency_offsets_squared + bandwidth_detection_pi_pulse**2)
        excitation_probabilities = (bandwidth_detection_pi_half_pulse / rabi_frequencies_pi_half_pulse) * \
            np.sin(2 * np.pi * rabi_frequencies_pi_half_pulse * self.detection_pi_half_pulse_length) * \
            0.25 * (bandwidth_detection_pi_pulse / rabi_frequencies_pi_pulse)**4 * \
            (1 - np.cos(2 * np.pi * rabi_frequencies_pi_pulse * self.detection_pi_pulse_length))**2
        detection_bandwidth = {"freq": frequency_axis, "prob": excitation_probabilities}
        return detection_bandwidth
    
    
    def compute_pump_bandwidth(self):
        """Compute the bandwidth of a pump pulse."""
        frequency_axis = np.arange(
            self.pump_frequency - 0.5 * self.pump_frequency_width, 
            self.pump_frequency + 0.5 * self.pump_frequency_width + self.frequency_increment, 
            self.frequency_increment
            )
        time_axis = np.arange(0, self.pump_pulse_length + self.time_increment, self.time_increment)
        frequency_axis_size = frequency_axis.size
        time_axis_size = time_axis.size
        maximal_rabi_frequency = np.sqrt(self.critical_adiabaticity * self.pump_frequency_width / self.pump_pulse_length)
        microwave_frequencies = self.pump_frequency - 0.5 * self.pump_frequency_width + \
            self.pump_frequency_width * time_axis / self.pump_pulse_length
        adiabaticity_array = np.zeros((frequency_axis_size, time_axis_size))
        for i in range(frequency_axis_size):
            for j in range(time_axis_size): 
                if self.pump_pulse_rise_time == 0:
                    rabi_frequency = maximal_rabi_frequency
                    rabi_frequency_derivative = 0
                else:
                    if time_axis[j] < self.pump_pulse_rise_time:
                        rabi_frequency = maximal_rabi_frequency * np.sin(0.5 * np.pi * time_axis[j] / self.pump_pulse_rise_time)
                        rabi_frequency_derivative = maximal_rabi_frequency * (0.5 * np.pi / self.pump_pulse_rise_time) * \
                            np.cos(0.5 * np.pi * time_axis[j] / self.pump_pulse_rise_time)
                    elif time_axis[j] > self.pump_pulse_length - self.pump_pulse_rise_time:
                        rabi_frequency = maximal_rabi_frequency * np.sin(0.5 * np.pi * (self.pump_pulse_length - time_axis[j]) / self.pump_pulse_rise_time)
                        rabi_frequency_derivative = maximal_rabi_frequency * (-0.5 * np.pi / self.pump_pulse_rise_time) * \
                            np.cos(0.5 * np.pi * (self.pump_pulse_length - time_axis[j]) / self.pump_pulse_rise_time)
                    else:
                        rabi_frequency = maximal_rabi_frequency
                        rabi_frequency_derivative = 0   
                frequency_offset = frequency_axis[i] - microwave_frequencies[j]
                frequency_offset_derivative = -self.pump_frequency_width / self.pump_pulse_length
                if rabi_frequency == 0 and frequency_offset == 0:
                    adiabaticity_value = 0
                else:
                    adiabaticity_value = (rabi_frequency**2 + frequency_offset**2)**1.5 / \
                        np.abs(rabi_frequency * frequency_offset_derivative - frequency_offset * rabi_frequency_derivative)
                adiabaticity_array[i][j] = adiabaticity_value      
        adiabaticities = np.amin(adiabaticity_array, axis=1)
        inversion_probabilities = 1 - np.exp(-0.5 * np.pi * adiabaticities)
        frequency_axis = np.append([frequency_axis[0] - self.frequency_increment], frequency_axis)
        inversion_probabilities = np.append([0], inversion_probabilities)
        frequency_axis = np.append(frequency_axis, [frequency_axis[-1] + self.frequency_increment])
        inversion_probabilities = np.append(inversion_probabilities, [0])
        pump_bandwidth = {"freq": frequency_axis, "prob": inversion_probabilities}
        return pump_bandwidth
    
    
    def get_detection_bandwidth(self, ranges=()):
        """Return the bandwidth of detection pulses."""
        if ranges == ():
            return self.detection_bandwidth
        else:
            indices = np.where(
                np.logical_and(self.detection_bandwidth["freq"] >= ranges[0], self.detection_bandwidth["freq"] <= ranges[1])
                )
            detection_bandwidth = {
                "freq": detection_bandwidth["freq"][indices], 
                "prob": detection_bandwidth["prob"][indices]
                }
            return detection_bandwidth 
    
    
    def get_pump_bandwidth(self, ranges=()):
        """Return the bandwidths of pump pulse."""
        if ranges == ():
            return self.pump_bandwidth
        else:
            indices = np.where(
                np.logical_and(self.pump_bandwidth["freq"] >= ranges[0], self.pump_bandwidth["freq"] <= ranges[1]))
            pump_bandwidth = {
                "freq": pump_bandwidth["freq"][indices],
                "prob": pump_bandwidth["prob"][indices]
                }
            return pump_bandwidth      
    
    
    def detection_probability(self, resonance_frequencies):
        """Compute detection probabilities for given resonance frequencies."""
        indices = find_nearest(self.detection_bandwidth["freq"], resonance_frequencies)
        detection_probabilities = self.detection_bandwidth["prob"][indices]
        return detection_probabilities
    
    
    def pump_probability(self, resonance_frequencies):
        """Compute pump probabilities for given resonance frequencies."""
        indices = find_nearest(self.pump_bandwidth["freq"], resonance_frequencies)
        pump_probabilities = self.pump_bandwidth["prob"][indices]
        return pump_probabilities