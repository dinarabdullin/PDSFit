'''Simulator class '''

import numpy as np
from scipy.spatial.transform import Rotation
import time
import datetime

from supplement.definitions import const
from spin_physics.spin import Spin
from mathematics.random_points_on_sphere import random_points_on_sphere
from mathematics.random_points_from_distribution import random_points_from_distribution
from mathematics.coordinate_system_conversions import spherical2cartesian, cartesian2spherical
from mathematics.rotate_coordinate_system import rotate_coordinate_system
from mathematics.histogram import histogram
from plots.plot_grids import plot_grids


class Simulator():

    def __init__(self, calculation_settings):
        self.integration_method = calculation_settings['integration_method']
        if self.integration_method == 'monte_carlo':
            self.mc_sample_size = calculation_settings['mc_sample_size']
            self.separate_grids = False
        else:
            self.grid_size = calculation_settings['grid_size']
            self.separate_grids = True
        self.distributions = calculation_settings['distributions']
        self.excitation_threshold = calculation_settings['excitation_treshold']
        self.frequency_increment_epr_spectrum = 0.001 # in GHz
        self.field_orientations = []
        self.weights_field_orientations = []
        self.euler_angles_convention = 'ZXZ'
        
    def set_field_orientations(self): 
        ''' Random points on a sphere '''
        return random_points_on_sphere(self.mc_sample_size) 
    
    def set_field_orientations_grid(self):
        ''' Powder-averging grid via Lebedev angular quadrature '''
    
    def set_r_values(self, r_mean, r_width, rel_prob):
        ''' Random points of r from a given distribution P(r) '''
        r_values = random_points_from_distribution(self.distributions['r'], r_mean, r_width, rel_prob, self.mc_sample_size)
        # Check that all r values are positive numbers 
        indices_nonpositive_r_values = np.argwhere(r_values <= 0).flatten()
        if indices_nonpositive_r_values.size == 0:
            return r_values
        else:
            for index in indices_nonpositive_r_values:
                while True:
                    r_value = random_points_from_distribution(self.distributions['r'], r_mean, r_width, rel_prob, 1)
                    if r_value > 0:
                        r_values[index] = r_value
                        break
            return r_values

    def set_r_values_grid(self, r_mean, r_width, rel_prob):
        ''' r-grid via Gauss-Legendre quadrature '''
    
    def set_j_values(self, j_mean, j_width, rel_prob):
        ''' Random points of j from a given distribution P(j) '''
        j_values = random_points_from_distribution(self.distributions['j'], j_mean, j_width, rel_prob, self.mc_sample_size)
        return j_values

    def set_j_values_grid(self, j_mean, j_width, rel_prob):
        ''' j-grid via Gauss-Legendre quadrature '''
    
    def set_r_orientations(self, xi_mean, xi_width, phi_mean, phi_width, rel_prob):
        ''' Random points of xi and phi from the corresponding distributions P(xi) and P(phi).
        They are used to compute the orientations of a distance vector in a reference frame '''
        xi_values = random_points_from_distribution(self.distributions['xi'], xi_mean, xi_width, rel_prob, self.mc_sample_size)
        phi_values = random_points_from_distribution(self.distributions['phi'], phi_mean, phi_width, rel_prob, self.mc_sample_size)
        xi_size = xi_values.size
        total_size = max([xi_values.size, phi_values.size])
        if xi_values.size < total_size:
            num_repeat = total_size / xi_values.size
            xi_values = np.repeat(xi_values, num_repeat)
        elif phi_values.size < total_size:
            num_repeat = total_size / phi_values.size
            phi_values = np.repeat(phi_values, num_repeat)
        r_orientations = spherical2cartesian(np.ones(total_size), xi_values, phi_values)
        if xi_size == 1:
            weights_r_orientations = np.ones(total_size)
        else:
            weights_r_orientations = np.sin(xi_values)
        return r_orientations, weights_r_orientations
    
    def set_r_orientations_grid(self, xi_mean, xi_width, phi_mean, phi_width, rel_prob):
        ''' xi/phi-grid via Lebedev angular quadrature.
        It is  used to compute the orientations of a distance vector in a reference frame '''

    def set_spin_frame_rotations(self, alpha_mean, alpha_width, beta_mean, beta_width, gamma_mean, gamma_width, rel_prob):
        ''' Random points of alpha, beta, and gamma from the corresponding distributions P(alpha), P(beta), and P(gamma).
        They are used to compute rotation matrices transforming a reference frame into a spin frame '''
        alpha_values = random_points_from_distribution(self.distributions['alpha'], alpha_mean, alpha_width, rel_prob, self.mc_sample_size)
        beta_values = random_points_from_distribution(self.distributions['beta'], beta_mean, beta_width, rel_prob, self.mc_sample_size)
        gamma_values = random_points_from_distribution(self.distributions['gamma'], gamma_mean, gamma_width, rel_prob, self.mc_sample_size)
        beta_size = beta_values.size
        total_size = max([alpha_values.size, beta_values.size, gamma_values.size])
        if alpha_values.size < total_size:
            num_repeat = total_size / alpha_values.size
            alpha_values = np.repeat(alpha_values, num_repeat)
        if beta_values.size < total_size:
            num_repeat = total_size / beta_values.size
            beta_values = np.repeat(beta_values, num_repeat)
        if gamma_values.size < total_size:
            num_repeat = total_size / gamma_values.size
            gamma_values = np.repeat(gamma_values, num_repeat)
        spin_frame_rotations = Rotation.from_euler(self.euler_angles_convention, np.column_stack((alpha_values, beta_values, gamma_values)))
        if beta_size == 1:
            weigths_spin_frame_rotations = np.ones(total_size)
        else:
            weigths_spin_frame_rotations = np.sin(beta_values)
        return spin_frame_rotations, weigths_spin_frame_rotations
    
    def set_spin_frame_rotations_grid(self, alpha_mean, alpha_width, beta_mean, beta_width, gamma_mean, gamma_width, rel_prob):
        ''' alpha/beta/gamma-grid via Mitchell grid.
        It is used to compute rotation matrices transforming a reference frame into a spin frame '''   
    
    def time_trace_from_dipolar_frequencies(self, experiment, modulation_frequencies, modulation_depths):
        ''' Converts dipolar frequencies into a PDS time trace '''
        simulated_time_trace = {}
        simulated_time_trace['t'] = experiment.t
        num_time_points = experiment.t.size
        simulated_time_trace['s'] = np.ones(num_time_points)
        for i in range(num_time_points):
            simulated_time_trace['s'][i] -= np.sum(modulation_depths * (1.0 - np.cos(2*np.pi * modulation_frequencies * experiment.t[i])))
        return simulated_time_trace
    
    def time_trace_from_dipolar_spectrum(self, experiment, modulation_frequencies, modulation_depths):
        ''' Converts a dipolar spectrum into a PDS time trace '''
        new_modulation_frequencies = np.arange(np.amin(modulation_frequencies), np.amax(modulation_frequencies), 0.01)
        new_modulation_depths = histogram(modulation_frequencies, bins=new_modulation_frequencies, weights=modulation_depths)
        simulated_time_trace = {}
        simulated_time_trace['t'] = experiment.t
        num_time_points = experiment.t.size
        simulated_time_trace['s'] = np.ones(num_time_points)
        for i in range(num_time_points):
            simulated_time_trace['s'][i] -= np.sum(new_modulation_depths * (1.0 - np.cos(2*np.pi * new_modulation_frequencies * experiment.t[i])))
        return simulated_time_trace
    
    def compute_time_trace_via_monte_carlo(self, experiment, spins, variables):
        ''' Computes a PDS time trace via Monte-Carlo integration '''
        if len(spins) == 2:
            time_start = time.time()
            # Random orientations of the applied static magnetic field in a reference frame
            if self.field_orientations == []:
                self.field_orientations = self.set_field_orientations()
            # Distance values
            r_values = self.set_r_values(variables['r_mean'][0], variables['r_width'][0], variables['rel_prob'][0])
            # Exchange coupling values
            j_values = self.set_j_values(variables['j_mean'][0], variables['j_width'][0], variables['rel_prob'][0])
            # The orientations of a distance vector in a reference frame and corresponding weights
            r_orientations, weights_r_orientations = self.set_r_orientations(variables['xi_mean'][0], variables['xi_width'][0], 
                variables['phi_mean'][0], variables['phi_width'][0], variables['rel_prob'][0])
            # Rotation matrices transforming a reference frame into a spin frame
            spin_frame_rotations, weigths_spin_frame_rotations = self.set_spin_frame_rotations(variables['alpha_mean'][0], variables['alpha_width'][0], 
                variables['beta_mean'][0], variables['beta_width'][0], variables['gamma_mean'][0], variables['gamma_width'][0], variables['rel_prob'][0])
            print('Monte-Carlo samples: %s\n' % str(datetime.timedelta(seconds = time.time()-time_start)))
            # Plot Monte-Carlo grids
            rho_values, xi_values, phi_values = cartesian2spherical(r_orientations)
            euler_angles = spin_frame_rotations.as_euler(self.euler_angles_convention, degrees=False)
            alpha_values, beta_values, gamma_values = euler_angles[:,0], euler_angles[:,1], euler_angles[:,2]
            # plot_grids(r_values, [], j_values, [], xi_values, weights_r_orientations, phi_values, [],
                       # alpha_values, [], beta_values, weigths_spin_frame_rotations, gamma_values, [])
            plot_grids(r_values, [], j_values, [], xi_values, [], phi_values, [],
                       alpha_values, [], beta_values, [], gamma_values, [])
            time_start = time.time()
            # Orientations of the applied static magnetic field in both spin frames
            field_orientations_spin1 = self.field_orientations
            field_orientations_spin2 = rotate_coordinate_system(self.field_orientations, spin_frame_rotations, self.separate_grids)
            # Resonance frequencies of both spins
            resonance_frequencies_spin1, effective_gfactors_spin1 = spins[0].res_freq(field_orientations_spin1, experiment.magnetic_field)
            resonance_frequencies_spin2, effective_gfactors_spin2 = spins[1].res_freq(field_orientations_spin2, experiment.magnetic_field)
            print('Resonance frequencies: %s\n' % str(datetime.timedelta(seconds = time.time() - time_start)))
            time_start = time.time()
            # Detection probabilities
            detection_probabilities_spin1 = experiment.detection_probability(resonance_frequencies_spin1, spins[0].int_res_freq)
            detection_probabilities_spin2 = experiment.detection_probability(resonance_frequencies_spin2, spins[1].int_res_freq)
            # Pump probabilities
            if experiment.technique == 'peldor':
                pump_probabilities_spin1 = np.where(detection_probabilities_spin2 > self.excitation_threshold,
                                                    experiment.pump_probability(resonance_frequencies_spin1, spins[0].int_res_freq), 0.0)
                pump_probabilities_spin2 = np.where(detection_probabilities_spin1 > self.excitation_threshold,
                                                    experiment.pump_probability(resonance_frequencies_spin2, spins[1].int_res_freq), 0.0)   
            elif experiment.technique == 'ridme':
                pump_probabilities_spin1 = np.where(detection_probabilities_spin2 > self.excitation_threshold,
                                                    experiment.pump_probability(spins[0].T1, spins[0].gAnisotropy, effective_gfactors_spin1), 0.0)
                pump_probabilities_spin2 = np.where(detection_probabilities_spin1 > self.excitation_threshold,                                    
                                                    experiment.pump_probability(spins[1].T1, spins[1].gAnisotropy, effective_gfactors_spin2), 0.0)
            indices_nonzero_probabilities = np.argwhere(np.logical_or(pump_probabilities_spin1 > self.excitation_threshold, pump_probabilities_spin2 > self.excitation_threshold)).flatten()
            detection_probabilities_spin1 = detection_probabilities_spin1[indices_nonzero_probabilities]
            detection_probabilities_spin2 = detection_probabilities_spin2[indices_nonzero_probabilities]
            pump_probabilities_spin1 = pump_probabilities_spin1[indices_nonzero_probabilities]
            pump_probabilities_spin2 = pump_probabilities_spin2[indices_nonzero_probabilities]
            print('Detection/pump probabilities: %s\n' % str(datetime.timedelta(seconds = time.time() - time_start)))
            time_start = time.time()
            # Modulation depths
            if weights_r_orientations.size == self.mc_sample_size:
                weights_r_orientations = weights_r_orientations[indices_nonzero_probabilities]
            if weigths_spin_frame_rotations.size == self.mc_sample_size:
                weigths_spin_frame_rotations = weigths_spin_frame_rotations[indices_nonzero_probabilities]
            amplitudes = (detection_probabilities_spin1 + detection_probabilities_spin2) * weights_r_orientations * weigths_spin_frame_rotations
            modulation_amplitudes = (detection_probabilities_spin1 * pump_probabilities_spin2 + detection_probabilities_spin2 * pump_probabilities_spin1) * \
                                     weights_r_orientations * weigths_spin_frame_rotations
            modulation_depths = modulation_amplitudes / np.sum(amplitudes)
            print('Modulation depths: %s\n' % str(datetime.timedelta(seconds = time.time() - time_start)))
            time_start = time.time()
            # Dipolar frequencies
            if r_values.shape[0] == self.mc_sample_size:
                r_values = r_values[indices_nonzero_probabilities]
            if r_orientations.shape[0] == self.mc_sample_size:
                r_orientations = r_orientations[indices_nonzero_probabilities]
            field_orientations = self.field_orientations[indices_nonzero_probabilities]
            effective_gfactors_spin1 = effective_gfactors_spin1[indices_nonzero_probabilities]
            effective_gfactors_spin2 = effective_gfactors_spin2[indices_nonzero_probabilities]
            if not spins[0].g_anisotropy_in_dipolar_coupling and not spins[1].g_anisotropy_in_dipolar_coupling:
                angular_term = 1.0 - 3.0 * np.sum(r_orientations * field_orientations, axis=1)**2
                dipolar_frequencies = const['Fdd'] * effective_gfactors_spin1 * effective_gfactors_spin2 * angular_term / r_values**3
                modulation_frequencies = dipolar_frequencies + j_values
            # elif spins[0].g_anisotropy_in_dipolar_coupling and not spins[1].g_anisotropy_in_dipolar_coupling:
                # ...
            # elif not spins[0].g_anisotropy_in_dipolar_coupling and spins[1].g_anisotropy_in_dipolar_coupling:
                # ...
            # elif spins[0].g_anisotropy_in_dipolar_coupling and spins[1].g_anisotropy_in_dipolar_coupling:
                # ...
            print('Modulation frequencies: %s\n' % str(datetime.timedelta(seconds = time.time() - time_start)))
            time_start = time.time()
            # Time trace
            #simulated_time_trace = self.time_trace_from_dipolar_frequencies(experiment, modulation_frequencies, modulation_depths)
            simulated_time_trace = self.time_trace_from_dipolar_spectrum(experiment, modulation_frequencies, modulation_depths)
            print('Time trace: %s\n' % str(datetime.timedelta(seconds = time.time() - time_start)))
            time_start = time.time()
        #elif len(spins) == 3:
            # ...
        return simulated_time_trace
     
    def compute_time_trace_via_grids(self, experiment, spins, variables):
        ''' Computes a PDS time trace via integration grids '''
    
    def compute_time_trace(self, experiment, spins, variables):
        ''' Computes a PDS time trace for a given set of variables '''
        simulated_time_trace = {}
        if self.integration_method == 'monte_carlo':
            simulated_time_trace = self.compute_time_trace_via_monte_carlo(experiment, spins, variables)
        # elif self.integration_method == 'grids':
            # simulated_time_trace = self.compute_time_trace_via_grids(experiment, spins, variables)
        return simulated_time_trace   
   
    def epr_spectrum(self, spins, field_value):
        ''' Computes an EPR spectrum of a spin system '''
        # Random orientations of the static magnetic field
        if self.integration_method == 'monte_carlo':
            if self.field_orientations == []:
                self.field_orientations = self.set_field_orientations()
                self.weights_field_orientations = np.ones(self.field_orientations.shape[0])
        # elif self.integration_method == 'grids':
            # if self.field_orientations == []:
                # self.field_orientations, self.weights_field_orientations = set_field_orientations_grid()
        # Resonance frequencies and their probabilities
        all_frequencies = []
        all_probabilities = []
        for spin in spins:
            # Resonance frequencies
            resonance_frequencies, effective_gvalues = spin.res_freq(self.field_orientations, field_value)
            num_field_orientations = self.field_orientations.shape[0]
            weights = self.weights_field_orientations.reshape(num_field_orientations,1) * spin.int_res_freq
            # Frequency ranges
            min_resonance_frequency = np.amin(resonance_frequencies)
            max_resonance_frequency = np.amax(resonance_frequencies)
            # Spectrum
            frequencies = np.arange(np.around(min_resonance_frequency, 3), np.around(max_resonance_frequency)+self.frequency_increment_epr_spectrum, self.frequency_increment_epr_spectrum)
            probabilities = histogram(resonance_frequencies, bins=frequencies, weights=weights)
            all_frequencies.extend(frequencies)
            all_probabilities.extend(probabilities)
        min_frequency = np.amin(all_frequencies) - 0.150
        max_frequency = np.amax(all_frequencies) + 0.150
        spectrum = {}
        spectrum['f'] = np.arange(min_frequency, max_frequency+self.frequency_increment_epr_spectrum, self.frequency_increment_epr_spectrum)
        spectrum['s'] = histogram(all_frequencies, bins=spectrum['f'], weights=all_probabilities)
        return spectrum
