import sys
import time
import datetime
import numpy as np
from scipy.spatial.transform import Rotation
from simulation.simulator import Simulator
from mathematics.random_points_on_sphere import random_points_on_sphere
from mathematics.random_points_from_distribution import random_points_from_distribution
from mathematics.coordinate_system_conversions import spherical2cartesian, cartesian2spherical
from mathematics.histogram import histogram
from mathematics.chi2 import chi2
from mathematics.fft import fft
from supplement.definitions import const


class MonteCarloSimulator(Simulator):
    ''' Monte-Carlo Simulation class '''
    
    def __init__(self):
        super().__init__()
        self.parameter_names = {
            'number_of_montecarlo_samples': 'int_list'
        }
        self.frequency_increment_epr_spectrum = 0.001 # in GHz
        self.frequency_increment_dipolar_spectrum = {} # in MHz
        self.minimal_r_value = 1.5 # minimal distance in nm
        self.field_orientations = []
        self.effective_gfactors_spin1 = []
        self.detection_probabilities_spin1 = {}
        self.pump_probabilities_spin1 = {}      
    
    def set_calculation_settings(self, calculation_settings):
        ''' Set calculation settings '''
        self.number_of_montecarlo_samples = calculation_settings['number_of_montecarlo_samples']
        self.num_samples = self.number_of_montecarlo_samples[0]
        self.distribution_types = calculation_settings['distribution_types']
        self.excitation_threshold = calculation_settings['excitation_threshold']
        self.euler_angles_convention = calculation_settings['euler_angles_convention']
        
    def reset_num_samples(self):
        ''' Resets the number of Monte-Carlo samples '''
        if len(self.number_of_montecarlo_samples) == 2 and self.number_of_montecarlo_samples[1] != self.number_of_montecarlo_samples[0]:
            self.num_samples = self.number_of_montecarlo_samples[1]
            return True
        else:
            return False
        
    def set_background_model(self, background):
        ''' Set the background model and parameters '''
        self.background = background
    
    def run_precalculations(self, experiments, spins):
        ''' Computes the detection and pump probabilities for spin 1'''
        # Random orientations of the static magnetic field
        self.field_orientations = self.set_field_orientations()
        # # Plot orientations of the magnetic field
        # rho_values, xi_values, phi_values = cartesian2spherical(self.field_orientations)
        # from plots.monte_carlo.plot_monte_carlo_points import plot_monte_carlo_points
        # plot_monte_carlo_points([], xi_values, phi_values, [], [], [], [], 'magnetic_field_orientations.png')
        # Orientations of the applied static magnetic field in the frame of spin 1
        field_orientations_spin1 = self.field_orientations 
        for experiment in experiments:
            self.frequency_increment_dipolar_spectrum[experiment.name] = 0.5 / np.amax(experiment.t)
            # Resonance frequencies and effective g-values of spin 1
            resonance_frequencies_spin1, self.effective_gfactors_spin1 = spins[0].res_freq(field_orientations_spin1, experiment.magnetic_field)
            # Detection probabilities
            self.detection_probabilities_spin1[experiment.name] = experiment.detection_probability(resonance_frequencies_spin1)
            # Pump probabilities
            if experiment.technique == 'peldor': 
                self.pump_probabilities_spin1[experiment.name] = experiment.pump_probability(resonance_frequencies_spin1) 
            elif experiment.technique == 'ridme':
                self.pump_probabilities_spin1[experiment.name] = experiment.pump_probability(spins[0].T1, spins[0].g_anisotropy_in_dipolar_coupling, self.effective_gfactors_spin1)
    
    def simulate_epr_spectra(self, spins, experiments):
        ''' Computes the EPR spectrum of a spin system at multiple magnetic fields '''
        epr_spectra = []
        for experiment in experiments:
            epr_spectrum = self.simulate_epr_spectrum(spins, experiment.magnetic_field)
            epr_spectra.append(epr_spectrum)
        return epr_spectra
    
    def simulate_epr_spectrum(self, spins, field_value):
        ''' Computes the EPR spectrum of a spin system at a single magnetic field '''
        # Random orientations of the static magnetic field
        if self.field_orientations == []:
            self.field_orientations = self.set_field_orientations()  
        # Resonance frequencies and their probabilities
        all_frequencies = []
        all_probabilities = []
        for spin in spins:
            # Resonance frequencies
            resonance_frequencies, effective_gvalues = spin.res_freq(self.field_orientations, field_value)
            num_field_orientations = self.field_orientations.shape[0]
            weights = np.tile(spin.int_res_freq, (num_field_orientations,1))
            # Frequency ranges
            min_resonance_frequency = np.amin(resonance_frequencies)
            max_resonance_frequency = np.amax(resonance_frequencies)
            # Spectrum
            frequencies = np.arange(min_resonance_frequency, max_resonance_frequency+self.frequency_increment_epr_spectrum, self.frequency_increment_epr_spectrum)
            probabilities = histogram(resonance_frequencies, bins=frequencies, weights=weights)
            all_frequencies.extend(frequencies)
            all_probabilities.extend(probabilities)
        all_frequencies = np.array(all_frequencies)
        all_probabilities = np.array(all_probabilities)
        min_frequency = np.amin(all_frequencies) - 0.100
        max_frequency = np.amax(all_frequencies) + 0.100
        spectrum = {}
        spectrum['f'] = np.arange(min_frequency, max_frequency+self.frequency_increment_epr_spectrum, self.frequency_increment_epr_spectrum)
        spectrum['p'] = histogram(all_frequencies, bins=spectrum['f'], weights=all_probabilities)       
        return spectrum
        
    def simulate_bandwidths(self, experiments):
        ''' Computes the bandwidths of detection and pump pulses '''
        bandwidths = []
        for experiment in experiments:
            if experiment.technique == 'peldor':
                bandwidths_single_experiment = {}
                bandwidths_single_experiment['detection_bandwidth'] = experiment.get_detection_bandwidth()
                bandwidths_single_experiment['pump_bandwidth'] = experiment.get_pump_bandwidth()
                bandwidths.append(bandwidths_single_experiment)
            elif experiment.technique == 'ridme':
                bandwidths_single_experiment = {}
                bandwidths_single_experiment['detection_bandwidth'] = experiment.get_detection_bandwidth()
                bandwidths.append(bandwidths_single_experiment)
        return bandwidths   

    def simulate_time_traces(self, experiments, spins, model_parameters, reset_field_orientations=False, more_output=True, display_messages=True):
        ''' Simulates PDS time traces for a given geometric model '''
        if display_messages:
            sys.stdout.write('\n########################################################################\
                              \n#                              Simulation                              #\
                              \n########################################################################\n')
        if more_output:
            simulated_time_traces = []
            background_parameters = []
            background_time_traces = []
            background_free_time_traces = []
            simulated_spectra = []
            modulation_depths = []
            dipolar_angle_distributions = []
            for experiment in experiments:
                simulated_time_trace, background_parameters_single_time_trace, background_time_trace, \
                background_free_time_trace, simulated_spectrum, modulation_depth, dipolar_angle_distribution = \
                    self.simulate_time_trace(experiment, spins, model_parameters, reset_field_orientations, more_output, display_messages)
                simulated_time_traces.append(simulated_time_trace)
                background_parameters.append(background_parameters_single_time_trace)
                background_time_traces.append(background_time_trace)
                background_free_time_traces.append(background_free_time_trace)
                simulated_spectra.append(simulated_spectrum)
                modulation_depths.append(modulation_depth)
                dipolar_angle_distributions.append(dipolar_angle_distribution)
            return simulated_time_traces, background_parameters, background_time_traces, background_free_time_traces, simulated_spectra, modulation_depths, dipolar_angle_distributions
        else: 
            simulated_time_traces = []
            for experiment in experiments:
                simulated_time_trace = self.simulate_time_trace(experiment, spins, model_parameters, reset_field_orientations, more_output, display_messages)
                simulated_time_traces.append(simulated_time_trace)
            return simulated_time_traces

    def simulate_time_trace(self, experiment, spins, model_parameters, reset_field_orientations=False, more_output=True, display_messages=False):
        ''' Simulates a PDS time trace for a given geometric model '''
        if display_messages:
            sys.stdout.write('\nSimulating the PDS time trace of experiment \'{0}\'... '.format(experiment.name))
            sys.stdout.flush()
        if more_output:
            simulated_time_trace, background_parameters, background_time_trace, background_free_time_trace, simulated_spectrum, modulation_depth, dipolar_angle_distribution = \
                self.compute_time_trace(experiment, spins, model_parameters, reset_field_orientations, more_output, display_messages=False)
            # Display statistics
            if display_messages:
                sys.stdout.write('done!\n') 
                sys.stdout.write('Background parameters:\n') 
                for parameter_name in self.background.parameter_full_names:
                    sys.stdout.write('{0}: {1:<15.6f}\n'.format(self.background.parameter_full_names[parameter_name], background_parameters[parameter_name]))
                chi2_value = chi2(simulated_time_trace['s'], experiment.s, experiment.noise_std)
                sys.stdout.write('Chi-squared: {0:<15.1f}\n'.format(chi2_value))
                sys.stdout.flush()                
            return simulated_time_trace, background_parameters, background_time_trace, background_free_time_trace, simulated_spectrum, modulation_depth, dipolar_angle_distribution
        else:
            simulated_time_trace = self.compute_time_trace(experiment, spins, model_parameters, reset_field_orientations, more_output, display_messages=False) 
            if display_messages:
                sys.stdout.write('done!\n')
            return simulated_time_trace

    def compute_time_trace(self, experiment, spins, model_parameters, reset_field_orientations=False, more_output=True, display_messages=False):
        ''' Computes a PDS time trace for a given geometric mode '''
        timings = [['Timings:', '']]
        time_start = time.time()
        # Random orientations of the applied static magnetic field in the reference frame
        if reset_field_orientations:
            self.field_orientations = []
            self.effective_gfactors_spin1 = []
            self.detection_probabilities_spin1 = {}
            self.pump_probabilities_spin1 = {}
        if self.field_orientations == []:
            self.field_orientations = self.set_field_orientations()
        # # Plot orientations of the magnetic field
        # rho_values, xi_values, phi_values = cartesian2spherical(self.field_orientations)
        # from plots.monte_carlo.plot_monte_carlo_points import plot_monte_carlo_points
        # plot_monte_carlo_points([], xi_values, phi_values, [], [], [], [], 'magnetic_field_orientations.png')
        # Distance values
        r_values = self.set_r_values(model_parameters['r_mean'], model_parameters['r_width'], model_parameters['rel_prob'])
        # Orientations of the distance vector in the reference frame
        r_orientations = self.set_r_orientations(model_parameters['xi_mean'], model_parameters['xi_width'], 
                                                 model_parameters['phi_mean'], model_parameters['phi_width'], 
                                                 model_parameters['rel_prob'])
        # Rotation matrices transforming the reference frame into the spin 2 frame
        spin_frame_rotations = self.set_spin_frame_rotations(model_parameters['alpha_mean'], model_parameters['alpha_width'], 
                                                                   model_parameters['beta_mean'], model_parameters['beta_width'], 
                                                                   model_parameters['gamma_mean'], model_parameters['gamma_width'], 
                                                                   model_parameters['rel_prob'])
        # Exchange coupling values
        j_values = self.set_j_values(model_parameters['j_mean'], model_parameters['j_width'], model_parameters['rel_prob'])                                                                                               
        timings.append(['Monte-Carlo samples (total)', str(datetime.timedelta(seconds = time.time()-time_start))])
        time_start = time.time()
        ## Plot Monte-Carlo points
        # rho_values, xi_values, phi_values = cartesian2spherical(r_orientations)
        # euler_angles = spin_frame_rotations.inv().as_euler(self.euler_angles_convention, degrees=False)
        # alpha_values, beta_values, gamma_values = euler_angles[:,0], euler_angles[:,1], euler_angles[:,2]
        # from plots.monte_carlo.plot_monte_carlo_points import plot_monte_carlo_points
        # plot_monte_carlo_points(r_values, xi_values, phi_values, alpha_values, beta_values, gamma_values, [])
        # Orientations of the applied static magnetic field in both spin frames
        field_orientations_spin1 = self.field_orientations
        field_orientations_spin2 = spin_frame_rotations.apply(self.field_orientations)
        # Resonance frequencies and/or effective g-values of both spins
        if self.effective_gfactors_spin1 == []:
            resonance_frequencies_spin1, effective_gfactors_spin1 = spins[0].res_freq(field_orientations_spin1, experiment.magnetic_field)
        else:
            effective_gfactors_spin1 = self.effective_gfactors_spin1
        resonance_frequencies_spin2, effective_gfactors_spin2 = spins[1].res_freq(field_orientations_spin2, experiment.magnetic_field)
        timings.append(['Resonance frequencies', str(datetime.timedelta(seconds = time.time()-time_start))])
        time_start = time.time()
        # Detection probabilities
        if self.detection_probabilities_spin1 == {}:
            detection_probabilities_spin1 = experiment.detection_probability(resonance_frequencies_spin1)
        else:
            detection_probabilities_spin1 = self.detection_probabilities_spin1[experiment.name]
        detection_probabilities_spin2 = experiment.detection_probability(resonance_frequencies_spin2)
        # Pump probabilities
        if experiment.technique == 'peldor': 
            if self.pump_probabilities_spin1 == {}:
                pump_probabilities_spin1 = experiment.pump_probability(resonance_frequencies_spin1)
            else:
                pump_probabilities_spin1 = self.pump_probabilities_spin1[experiment.name]
            pump_probabilities_spin2 = experiment.pump_probability(resonance_frequencies_spin2)
        elif experiment.technique == 'ridme':
            if self.pump_probabilities_spin1 == {}:
                pump_probabilities_spin1 = experiment.pump_probability(spins[0].T1, spins[0].g_anisotropy_in_dipolar_coupling, effective_gfactors_spin1)
            else:
                pump_probabilities_spin1 = self.pump_probabilities_spin1[experiment.name]
            if spins[0].num_res_freq > 1:
                pump_probabilities_spin1 = np.tile(pump_probabilities_spin1, spins[0].num_res_freq)
            pump_probabilities_spin2 = experiment.pump_probability(spins[1].T1, spins[1].g_anisotropy_in_dipolar_coupling, effective_gfactors_spin2)  
            if spins[1].num_res_freq > 1:
                pump_probabilities_spin2 = np.tile(pump_probabilities_spin2, spins[1].num_res_freq)
        # Take into account the weights of individual resonance frequencies and
        # the overlap of the pump and detection bandwidths
        weights_spin1 = spins[0].int_res_freq.reshape(spins[0].num_res_freq, 1)
        weights_spin2 = spins[1].int_res_freq.reshape(spins[1].num_res_freq, 1)
        corrected_detection_probabilities_spin1 = np.dot(detection_probabilities_spin1 * (1.0 - pump_probabilities_spin1), weights_spin1)
        corrected_detection_probabilities_spin2 = np.dot(detection_probabilities_spin2 * (1.0 - pump_probabilities_spin2), weights_spin2)
        corrected_pump_probabilities_spin1 = np.dot(pump_probabilities_spin1 * (1.0 - detection_probabilities_spin1), weights_spin1)
        corrected_pump_probabilities_spin2 = np.dot(pump_probabilities_spin2 * (1.0 - detection_probabilities_spin2), weights_spin2)
        timings.append(['Detection/pump probabilities', str(datetime.timedelta(seconds = time.time()-time_start))])
        time_start = time.time()
        # Modulation depths                                     
        modulation_depths = 1.0 / np.sum(corrected_detection_probabilities_spin1 + corrected_detection_probabilities_spin2) * \
                            (corrected_detection_probabilities_spin1 * corrected_pump_probabilities_spin2 + corrected_detection_probabilities_spin2 * corrected_pump_probabilities_spin1)            
        modulation_depths = modulation_depths.flatten()
        indices_nonzero_probabilities = np.where(modulation_depths > self.excitation_threshold)[0]
        modulation_depths = modulation_depths[indices_nonzero_probabilities]
        timings.append(['Modulation depths', str(datetime.timedelta(seconds = time.time()-time_start))])
        time_start = time.time()
        # Modulation frequencies
        field_orientations = self.field_orientations[indices_nonzero_probabilities]
        r_values = r_values[indices_nonzero_probabilities]
        r_orientations = r_orientations[indices_nonzero_probabilities]
        effective_gfactors_spin1 = effective_gfactors_spin1[indices_nonzero_probabilities]
        effective_gfactors_spin2 = effective_gfactors_spin2[indices_nonzero_probabilities]
        j_values = j_values[indices_nonzero_probabilities] 
        if not spins[0].g_anisotropy_in_dipolar_coupling and not spins[1].g_anisotropy_in_dipolar_coupling:
            angular_term = 1.0 - 3.0 * np.sum(r_orientations * field_orientations, axis=1)**2
        elif spins[0].g_anisotropy_in_dipolar_coupling and not spins[1].g_anisotropy_in_dipolar_coupling:
            quantization_axes_spin1 = spins[0].quantization_axis(field_orientations, effective_gfactors_spin1)
            angular_term = 1.0 - 3.0 * np.sum(r_orientations * quantization_axes_spin1, axis=1) * np.sum(r_orientations * field_orientations, axis=1)
        elif not spins[0].g_anisotropy_in_dipolar_coupling and spins[1].g_anisotropy_in_dipolar_coupling:
            field_orientations_spin2 = field_orientations_spin2[indices_nonzero_probabilities]
            spin_frame_rotations = spin_frame_rotations[indices_nonzero_probabilities]
            r_orientations_spin2 = spin_frame_rotations.apply(r_orientations)
            quantization_axes_spin2 = spins[1].quantization_axis(field_orientations_spin2, effective_gfactors_spin2)
            angular_term = 1.0 - 3.0 * np.sum(r_orientations * field_orientations, axis=1) * np.sum(r_orientations_spin2 * quantization_axes_spin2, axis=1) 
        elif spins[0].g_anisotropy_in_dipolar_coupling and spins[1].g_anisotropy_in_dipolar_coupling:
            quantization_axes_spin1 = spins[0].quantization_axis(field_orientations, effective_gfactors_spin1)
            field_orientations_spin2 = field_orientations_spin2[indices_nonzero_probabilities]
            spin_frame_rotations = spin_frame_rotations[indices_nonzero_probabilities]
            r_orientations_spin2 = spin_frame_rotations.apply(r_orientations)
            quantization_axes_spin2 = spins[1].quantization_axis(field_orientations_spin2, effective_gfactors_spin2)
            quantization_axes_spin2_ref = spin_frame_rotations.inv().apply(quantization_axes_spin2)
            angular_term = np.sum(quantization_axes_spin1 * quantization_axes_spin2_ref, axis=1) - \
                                  3.0 * np.sum(r_orientations * quantization_axes_spin1, axis=1) * np.sum(r_orientations_spin2 * quantization_axes_spin2, axis=1)
        dipolar_frequencies = const['Fdd'] * effective_gfactors_spin1 * effective_gfactors_spin2 * angular_term / r_values**3
        modulation_frequencies = dipolar_frequencies + j_values
        timings.append(['Dipolar frequencies', str(datetime.timedelta(seconds = time.time()-time_start))])
        time_start = time.time()
        # PDS time trace
        frequency_increment = self.frequency_increment_dipolar_spectrum[experiment.name]
        intramolecular_time_trace = self.intramolecular_time_trace_from_dipolar_spectrum(experiment.t, modulation_frequencies, modulation_depths, frequency_increment)
        background_parameters = self.background.optimize_parameters(experiment.t, experiment.s, intramolecular_time_trace)
        simulated_time_trace_tmp = self.background.get_fit(experiment.t, background_parameters, intramolecular_time_trace)
        simulated_time_trace_tmp = simulated_time_trace_tmp / np.amax(simulated_time_trace_tmp)
        simulated_time_trace = {}
        simulated_time_trace['t'] = experiment.t
        simulated_time_trace['s'] = simulated_time_trace_tmp
        timings.append(['PDS time trace', str(datetime.timedelta(seconds = time.time()-time_start))])
        # Display statistics
        if display_messages:
            sys.stdout.write('\nNumber of Monte-Carlo samples with non-zero weights: {0} out of {1}\n'.format(indices_nonzero_probabilities.size, self.num_samples))
            for instance in timings:
                sys.stdout.write('{:<30} {:<30}\n'.format(instance[0], instance[1]))
            sys.stdout.flush()
        if more_output:
            # Distribution of dipolar angles
            dipolar_angle_values = const['rad2deg'] * np.arccos(np.sum(field_orientations * r_orientations, axis=1))
            dipolar_angle_values = dipolar_angle_values.flatten()
            dipolar_angle_values = np.where(dipolar_angle_values < 0, -dipolar_angle_values, dipolar_angle_values)
            dipolar_angle_distribution = {}
            dipolar_angle_distribution['v'] = np.arange(0, 181, 1) 
            dipolar_angle_probabilities = histogram(dipolar_angle_values, bins=dipolar_angle_distribution['v'], weights=modulation_depths)
            dipolar_angle_probabilities = const['rad2deg'] * dipolar_angle_probabilities / sum(dipolar_angle_probabilities)
            dipolar_angle_distribution['p'] = dipolar_angle_probabilities
            # Background-free time trace
            total_modulation_depth = np.sum(modulation_depths)
            background_time_trace = {}
            background_time_trace['t'] = experiment.t
            background_time_trace['s'] = self.background.get_background(experiment.t, background_parameters, total_modulation_depth)
            background_free_time_trace = {}
            background_free_time_trace['t'] = experiment.t
            background_free_time_trace['s'] = simulated_time_trace['s']  * np.amax(background_time_trace['s']) / background_time_trace['s']
            background_free_time_trace['se'] = experiment.s * np.amax(background_time_trace['s']) / background_time_trace['s']
            # Dipolar spectrum
            dc_offset = 1 - total_modulation_depth * background_parameters['scale_factor']
            f, spc1 = fft(background_free_time_trace['t'], background_free_time_trace['s'], dc_offset = dc_offset)
            f, spc2 = fft(background_free_time_trace['t'], background_free_time_trace['se'], dc_offset = dc_offset)
            simulated_spectrum = {}
            simulated_spectrum['f'] = f
            simulated_spectrum['p'] = spc1
            simulated_spectrum['pe'] = spc2
            return simulated_time_trace, background_parameters, background_time_trace, background_free_time_trace, simulated_spectrum, total_modulation_depth, dipolar_angle_distribution
        else:
            return simulated_time_trace
    
    def set_field_orientations(self): 
        ''' Generates random samples on a sphere '''
        return random_points_on_sphere(self.num_samples) 
    
    def set_r_values(self, r_mean, r_width, rel_prob):
        ''' Generates random samples of r from the distribution P(r) '''
        r_values = random_points_from_distribution(self.distribution_types['r'], r_mean, r_width, rel_prob, self.num_samples, False)
        # Check that all r values are above the lower limit
        indices_r_values_below_limit = np.where(r_values < self.minimal_r_value)[0]
        if indices_r_values_below_limit.size == 0:
            return r_values
        else:
            for index in indices_r_values_below_limit:
                while True:
                    r_value = random_points_from_distribution(self.distribution_types['r'], r_mean, r_width, rel_prob, 1, False)
                    if r_value >= self.minimal_r_value:
                        r_values[index] = r_value
                        break
            return r_values

    def set_r_orientations(self, xi_mean, xi_width, phi_mean, phi_width, rel_prob):
        ''' 
        Generates random samples of xi and phi from the distributions P(xi)*sin(xi) and P(phi).
        Computes the orientations of the distance vector in the reference frame.
        '''
        xi_values = random_points_from_distribution(self.distribution_types['xi'], xi_mean, xi_width, rel_prob, self.num_samples, True)
        phi_values = random_points_from_distribution(self.distribution_types['phi'], phi_mean, phi_width, rel_prob, self.num_samples, False)
        r_orientations = spherical2cartesian(np.ones(self.num_samples), xi_values, phi_values)
        # # Plot Monte-Carlo points
        # from plots.monte_carlo.plot_monte_carlo_points import plot_monte_carlo_points
        # plot_monte_carlo_points([], xi_values, phi_values, [], [], [], [], "spherical_coordinates.png")
        return r_orientations    
    
    def set_spin_frame_rotations(self, alpha_mean, alpha_width, beta_mean, beta_width, gamma_mean, gamma_width, rel_prob):
        ''' 
        Generates random samples of alpha, beta, and gamma from the distributions P(alpha), P(beta)*sin(beta), and P(gamma).
        Compute the rotation matrices transforming the reference frame into the frame of spin 2, or 3, etc.
        '''
        alpha_values = random_points_from_distribution(self.distribution_types['alpha'], alpha_mean, alpha_width, rel_prob, self.num_samples, False)
        beta_values = random_points_from_distribution(self.distribution_types['beta'], beta_mean, beta_width, rel_prob, self.num_samples, True)
        gamma_values = random_points_from_distribution(self.distribution_types['gamma'], gamma_mean, gamma_width, rel_prob, self.num_samples, False)
        # Rotation matrix of intrinsic, passive rotations
        spin_frame_rotations = Rotation.from_euler(self.euler_angles_convention, np.column_stack((alpha_values, beta_values, gamma_values)))
        # Convert active rotations to passive rotations
        spin_frame_rotations = spin_frame_rotations.inv()
        # # Plot Monte-Carlo points
        # from plots.monte_carlo.plot_monte_carlo_points import plot_monte_carlo_points
        # plot_monte_carlo_points([], [], [], alpha_values, beta_values, gamma_values, [], "euler_angles.png")
        return spin_frame_rotations
        
    def set_j_values(self, j_mean, j_width, rel_prob):
        ''' Generates the randon samples of J from distribution P(J) '''
        if j_mean[0] == 0 and j_width[0] == 0:
            j_values = np.zeros(self.num_samples)
        else:
            j_values = random_points_from_distribution(self.distribution_types['j'], j_mean, j_width, [1.0], self.num_samples, False)
        return j_values

    def intramolecular_time_trace_from_dipolar_frequencies(self, t, modulation_frequencies, modulation_depths):
        ''' Computes the intramolecular part of a PDS time trace based on the dipolar frequencies '''
        num_time_points = t.size
        intramolecular_time_trace = np.ones(num_time_points)
        if modulation_frequencies.size != 0:
            for i in range(num_time_points):
                intramolecular_time_trace[i] -= np.sum(modulation_depths * (1 - np.cos(2*np.pi * modulation_frequencies * t[i])))
        return intramolecular_time_trace
    
    def intramolecular_time_trace_from_dipolar_spectrum(self, t, modulation_frequencies, modulation_depths, frequency_increment):
        ''' Computes the intramolecular part of a PDS time trace based on the dipolar spectrum '''
        num_time_points = t.size
        intramolecular_time_trace = np.ones(num_time_points)
        if modulation_frequencies.size != 0:
            modulation_frequency_min = np.amin(modulation_frequencies)
            modulation_frequency_max = np.amax(modulation_frequencies)
            if (modulation_frequency_max - modulation_frequency_min > frequency_increment):
                new_modulation_frequencies = np.arange(np.amin(modulation_frequencies), np.amax(modulation_frequencies), frequency_increment)
                new_modulation_depths = histogram(modulation_frequencies, bins=new_modulation_frequencies, weights=modulation_depths)
            else:
                new_modulation_frequencies = np.array([modulation_frequency_min])
                new_modulation_depths = np.array([np.sum(modulation_depths)])
            for i in range(num_time_points):
                intramolecular_time_trace[i] -= np.sum(new_modulation_depths * (1 - np.cos(2*np.pi * new_modulation_frequencies * t[i])))
        return intramolecular_time_trace