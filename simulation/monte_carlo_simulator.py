import sys
import time
import datetime
import numpy as np
from scipy.spatial.transform import Rotation
from simulation.simulator import Simulator
from mathematics.random_points_on_sphere import random_points_on_sphere
from mathematics.random_points_from_distribution import random_points_from_distribution, random_points_from_sine_weighted_distribution
from mathematics.coordinate_system_conversions import spherical2cartesian, cartesian2spherical
from mathematics.rotate_coordinate_system import rotate_coordinate_system
from mathematics.histogram import histogram
from mathematics.chi2 import chi2
from supplement.definitions import const


class MonteCarloSimulator(Simulator):
    ''' Monte-Carlo Simulation class '''
    
    def __init__(self, calculation_settings):
        super().__init__(calculation_settings)
        self.mc_sample_size = calculation_settings['mc_sample_size']
        self.separate_grids = False
        self.frequency_increment_epr_spectrum = 0.001 # in GHz
        self.field_orientations = []
        self.effective_gfactors_spin1 = []
        self.detection_probabilities_spin1 = {}
        self.pump_probabilities_spin1 = {}      
        
    def set_field_orientations(self): 
        ''' Random points on a sphere '''
        return random_points_on_sphere(self.mc_sample_size) 
    
    def set_r_values(self, r_mean, r_width, rel_prob):
        ''' Random points of r from distribution P(r) '''
        r_values = random_points_from_distribution(self.distributions['r'], r_mean, r_width, rel_prob, self.mc_sample_size)
        # Check that all r values are positive numbers 
        indices_nonpositive_r_values = np.where(r_values <= 0)[0]
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

    def set_r_orientations(self, xi_mean, xi_width, phi_mean, phi_width, rel_prob):
        ''' 
        Random points of xi and phi from corresponding distributions P(xi) and P(phi)
        are used to compute the orientations of the distance vector in the reference frame
        '''
        xi_values = random_points_from_sine_weighted_distribution(self.distributions['xi'], xi_mean, xi_width, rel_prob, self.mc_sample_size)
        phi_values = random_points_from_distribution(self.distributions['phi'], phi_mean, phi_width, rel_prob, self.mc_sample_size)
        r_orientations = spherical2cartesian(np.ones(self.mc_sample_size), xi_values, phi_values)
        return r_orientations
   
    def set_spin_frame_rotations(self, alpha_mean, alpha_width, beta_mean, beta_width, gamma_mean, gamma_width, rel_prob):
        '''
        Random points of alpha, beta, and gamma from corresponding distributions P(alpha), P(beta), and P(gamma)
        are used to compute rotation matrices transforming the reference frame into the spin frame
        '''
        alpha_values = random_points_from_distribution(self.distributions['alpha'], alpha_mean, alpha_width, rel_prob, self.mc_sample_size)
        beta_values = random_points_from_sine_weighted_distribution(self.distributions['beta'], beta_mean, beta_width, rel_prob, self.mc_sample_size)
        gamma_values = random_points_from_distribution(self.distributions['gamma'], gamma_mean, gamma_width, rel_prob, self.mc_sample_size)
        spin_frame_rotations = Rotation.from_euler(self.euler_angles_convention, np.column_stack((alpha_values, beta_values, gamma_values)))
        # Convert active rotations to passive rotations
        spin_frame_rotations = spin_frame_rotations.inv()
        return spin_frame_rotations
        
    def set_j_values(self, j_mean, j_width, rel_prob):
        ''' Random points of j from distribution P(j) '''
        j_values = random_points_from_distribution(self.distributions['j'], j_mean, j_width, rel_prob, self.mc_sample_size)
        return j_values
    
    def set_coordinates(self, r_mean, r_width, xi_mean, xi_width, phi_mean, phi_width, rel_prob):
        ''' 
        Random points of r, xi, and phi from corresponding distributions P(r), P(xi), and P(phi)
        are used to compute the coordinates of the distance vector in the reference frame
        '''
        r_values = self.set_r_values(r_mean, r_width, rel_prob)
        xi_values = random_points_from_sine_weighted_distribution(self.distributions['xi'], xi_mean, xi_width, rel_prob, self.mc_sample_size)
        phi_values = random_points_from_distribution(self.distributions['phi'], phi_mean, phi_width, rel_prob, self.mc_sample_size)
        coordinates = spherical2cartesian(r_values, xi_values, phi_values)
        return coordinates

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
        simulated_time_trace = {}
        simulated_time_trace['t'] = experiment.t
        num_time_points = experiment.t.size
        simulated_time_trace['s'] = np.ones(num_time_points)
        if modulation_frequencies.size != 0:
            modulation_frequency_min = np.amin(modulation_frequencies)
            modulation_frequency_max = np.amax(modulation_frequencies)
            if modulation_frequency_min != modulation_frequency_max:
                new_modulation_frequencies = np.arange(np.amin(modulation_frequencies), np.amax(modulation_frequencies), 0.01)
                new_modulation_depths = histogram(modulation_frequencies, bins=new_modulation_frequencies, weights=modulation_depths)
            else:
                new_modulation_frequencies = np.array([modulation_frequency_min])
                new_modulation_depths = np.array([np.sum(modulation_depths)])
            for i in range(num_time_points):
                simulated_time_trace['s'][i] -= np.sum(new_modulation_depths * (1.0 - np.cos(2*np.pi * new_modulation_frequencies * experiment.t[i])))
        return simulated_time_trace

    def rescale_modulation_depth(self, time_trace, current_modulation_depth, new_modulation_depth):
        ''' Rescales the modulation depth of a PDS time trace'''
        scale_factor = new_modulation_depth / current_modulation_depth
        if self.scale_range_modulation_depth != []:
            if scale_factor < self.scale_range_modulation_depth[0]:
                scale_factor = self.scale_range_modulation_depth[0]
            elif scale_factor > self.scale_range_modulation_depth[1]:
                scale_factor = self.scale_range_modulation_depth[1]
        new_time_trace = np.ones(time_trace.shape) - time_trace
        new_time_trace = scale_factor * new_time_trace
        new_time_trace = np.ones(time_trace.shape) - new_time_trace
        return new_time_trace, scale_factor
    
    def compute_time_trace_via_monte_carlo(self, experiment, spins, variables, display_messages=False):
        ''' Computes a PDS time trace via Monte-Carlo integration '''
        timings = [['Timings:', '']]
        time_start = time.time()
        # Random orientations of the applied static magnetic field in the reference frame
        if self.field_orientations == []:
            self.field_orientations = self.set_field_orientations()
        # Distance values
        r_values = self.set_r_values(variables['r_mean'][0], variables['r_width'][0], variables['rel_prob'][0])
        # Orientations of the distance vector in the reference frame
        r_orientations = self.set_r_orientations(variables['xi_mean'][0], variables['xi_width'][0], 
                                                 variables['phi_mean'][0], variables['phi_width'][0], 
                                                 variables['rel_prob'][0])
        # Rotation matrices transforming the reference frame into the spin 2 frame
        spin_frame_rotations_spin2 = self.set_spin_frame_rotations(variables['alpha_mean'][0], variables['alpha_width'][0], 
                                                                   variables['beta_mean'][0], variables['beta_width'][0], 
                                                                   variables['gamma_mean'][0], variables['gamma_width'][0], 
                                                                   variables['rel_prob'][0])
        # Exchange coupling values
        j_values = self.set_j_values(variables['j_mean'][0], variables['j_width'][0], variables['rel_prob'][0])                                                                                               
        timings.append(['Monte-Carlo samples', str(datetime.timedelta(seconds = time.time()-time_start))])
        time_start = time.time()
        # # Plot Monte-Carlo grids
        # from plots.plot_grids import plot_grids
        # rho_values, xi_values, phi_values = cartesian2spherical(r_orientations)
        # euler_angles = spin_frame_rotations_spin2.inv().as_euler(self.euler_angles_convention, degrees=False)
        # alpha_values, beta_values, gamma_values = euler_angles[:,0], euler_angles[:,1], euler_angles[:,2]
        # plot_grids(r_values, [], xi_values, [], phi_values, [], alpha_values, [], beta_values, [], gamma_values, [], j_values, [])
        # Orientations of the applied static magnetic field in both spin frames
        field_orientations_spin1 = self.field_orientations
        field_orientations_spin2 = rotate_coordinate_system(self.field_orientations, spin_frame_rotations_spin2, self.separate_grids)
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
            detection_probabilities_spin1 = experiment.detection_probability(resonance_frequencies_spin1, spins[0].int_res_freq)
        else:
            detection_probabilities_spin1 = self.detection_probabilities_spin1[experiment.name]
        detection_probabilities_spin2 = experiment.detection_probability(resonance_frequencies_spin2, spins[1].int_res_freq)
        # Pump probabilities
        if experiment.technique == 'peldor': 
            if self.pump_probabilities_spin1 == {}:
                pump_probabilities_spin1 = np.where(detection_probabilities_spin2 > self.excitation_threshold,
                                                    experiment.pump_probability(resonance_frequencies_spin1, spins[0].int_res_freq), 0.0)
            else:
                pump_probabilities_spin1 = np.where(detection_probabilities_spin2 > self.excitation_threshold,
                                                    self.pump_probabilities_spin1[experiment.name], 0.0)
            pump_probabilities_spin2 = np.where(detection_probabilities_spin1 > self.excitation_threshold,
                                                experiment.pump_probability(resonance_frequencies_spin2, spins[1].int_res_freq), 0.0)   
        elif experiment.technique == 'ridme':
            if self.pump_probabilities_spin1 == {}:
                pump_probabilities_spin1 = np.where(detection_probabilities_spin2 > self.excitation_threshold,
                                                    experiment.pump_probability(spins[0].T1, spins[0].g_anisotropy_in_dipolar_coupling, effective_gfactors_spin1), 0.0)
            else:
                pump_probabilities_spin1 = np.where(detection_probabilities_spin2 > self.excitation_threshold,
                                                    self.pump_probabilities_spin1[experiment.name], 0.0)
            pump_probabilities_spin2 = np.where(detection_probabilities_spin1 > self.excitation_threshold,                                    
                                                experiment.pump_probability(spins[1].T1, spins[1].g_anisotropy_in_dipolar_coupling, effective_gfactors_spin2), 0.0)   
        indices_nonzero_probabilities_spin1 = np.where(pump_probabilities_spin1 > self.excitation_threshold)[0]
        indices_nonzero_probabilities_spin2 = np.where(pump_probabilities_spin2 > self.excitation_threshold)[0]
        indices_nonzero_probabilities = np.unique(np.concatenate((indices_nonzero_probabilities_spin1, indices_nonzero_probabilities_spin2), axis=None))
        indices_nonzero_probabilities = np.sort(indices_nonzero_probabilities, axis=None)
        detection_probabilities_spin1 = detection_probabilities_spin1[indices_nonzero_probabilities]
        detection_probabilities_spin2 = detection_probabilities_spin2[indices_nonzero_probabilities]
        pump_probabilities_spin1 = pump_probabilities_spin1[indices_nonzero_probabilities]
        pump_probabilities_spin2 = pump_probabilities_spin2[indices_nonzero_probabilities]
        timings.append(['Detection/pump probabilities', str(datetime.timedelta(seconds = time.time()-time_start))])
        time_start = time.time()
        # Modulation depths
        amplitudes = detection_probabilities_spin1 + detection_probabilities_spin2
        modulation_amplitudes = detection_probabilities_spin1 * pump_probabilities_spin2 + \
                                detection_probabilities_spin2 * pump_probabilities_spin1
        modulation_depths = modulation_amplitudes / np.sum(amplitudes)
        total_modulation_depth = np.sum(modulation_depths)
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
            angular_term = 1.0 - 3.0 * np.sum(r_orientations * quantization_axes_spin1, axis=1) * \
                                       np.sum(r_orientations * field_orientations, axis=1)
        elif not spins[0].g_anisotropy_in_dipolar_coupling and spins[1].g_anisotropy_in_dipolar_coupling:
            field_orientations_spin2 = field_orientations_spin2[indices_nonzero_probabilities]
            spin_frame_rotations_spin2 = spin_frame_rotations_spin2[indices_nonzero_probabilities]
            r_orientations_spin2 = rotate_coordinate_system(r_orientations, spin_frame_rotations_spin2, self.separate_grids)
            quantization_axes_spin2 = spins[1].quantization_axis(field_orientations_spin2, effective_gfactors_spin2)
            angular_term = 1.0 - 3.0 * np.sum(r_orientations * field_orientations, axis=1) * \
                                       np.sum(r_orientations_spin2 * quantization_axes_spin2, axis=1) 
        elif spins[0].g_anisotropy_in_dipolar_coupling and spins[1].g_anisotropy_in_dipolar_coupling:
            quantization_axes_spin1 = spins[0].quantization_axis(field_orientations, effective_gfactors_spin1)
            field_orientations_spin2 = field_orientations_spin2[indices_nonzero_probabilities]
            spin_frame_rotations_spin2 = spin_frame_rotations_spin2[indices_nonzero_probabilities]
            r_orientations_spin2 = rotate_coordinate_system(r_orientations, spin_frame_rotations_spin2, self.separate_grids)
            quantization_axes_spin2 = spins[1].quantization_axis(field_orientations_spin2, effective_gfactors_spin2)
            quantization_axes_spin2_ref = rotate_coordinate_system(quantization_axes_spin2, spin_frame_rotations_spin2.inv(), self.separate_grids)
            angular_term = np.sum(quantization_axes_spin1 * quantization_axes_spin2_ref, axis=1) - \
                                  3.0 * np.sum(r_orientations * quantization_axes_spin1, axis=1) * \
                                        np.sum(r_orientations_spin2 * quantization_axes_spin2, axis=1)
        dipolar_frequencies = const['Fdd'] * effective_gfactors_spin1 * effective_gfactors_spin2 * angular_term / r_values**3
        modulation_frequencies = dipolar_frequencies + j_values
        timings.append(['Dipolar frequencies', str(datetime.timedelta(seconds = time.time()-time_start))])
        time_start = time.time()
        # Time trace
        #simulated_time_trace = self.time_trace_from_dipolar_frequencies(experiment, modulation_frequencies, modulation_depths)
        simulated_time_trace = self.time_trace_from_dipolar_spectrum(experiment, modulation_frequencies, modulation_depths)
        timings.append(['PDS time trace', str(datetime.timedelta(seconds = time.time()-time_start))])
        # Messages
        if display_messages:
            print('Number of Monte-Carlo samples with non-zero weights: {0} out of {1}'.format(indices_nonzero_probabilities.size, self.mc_sample_size))
            for instance in timings:
                print('{:<30} {:<30}'.format(instance[0], instance[1]))
        return simulated_time_trace, total_modulation_depth
    
    def compute_time_trace_via_monte_carlo_multispin(self, experiment, spins, variables, idx_spin1=0, idx_spin2=1, display_messages=False):
        ''' Computes a PDS time trace via Monte-Carlo integration (multi-spin version) '''
        if idx_spin1 == 0:
            timings = [['Timings:', '']]
            time_start = time.time()
            # Random orientations of the applied static magnetic field in the reference frame
            if self.field_orientations == []:
                self.field_orientations = self.set_field_orientations()
            # Distance values
            r_values = self.set_r_values(variables['r_mean'][idx_spin2-1], variables['r_width'][idx_spin2-1], variables['rel_prob'][idx_spin2-1])
            # Orientations of the distance vector in the reference frame
            r_orientations = self.set_r_orientations(variables['xi_mean'][idx_spin2-1], variables['xi_width'][idx_spin2-1], 
                                                     variables['phi_mean'][idx_spin2-1], variables['phi_width'][idx_spin2-1], 
                                                     variables['rel_prob'][idx_spin2-1])
            # Rotation matrices transforming the reference frame into the spin 2 frame
            spin_frame_rotations_spin2 = self.set_spin_frame_rotations(variables['alpha_mean'][idx_spin2-1], variables['alpha_width'][idx_spin2-1], 
                                                                       variables['beta_mean'][idx_spin2-1], variables['beta_width'][idx_spin2-1], 
                                                                       variables['gamma_mean'][idx_spin2-1], variables['gamma_width'][idx_spin2-1], 
                                                                       variables['rel_prob'][idx_spin2-1])                                                                                               
            timings.append(['Monte-Carlo samples', str(datetime.timedelta(seconds = time.time()-time_start))])
            time_start = time.time()
            # Orientations of the applied static magnetic field in both spin frames
            field_orientations_spin1 = self.field_orientations
            field_orientations_spin2 = rotate_coordinate_system(self.field_orientations, spin_frame_rotations_spin2, self.separate_grids)
            # Resonance frequencies and/or effective g-values of both spins
            if self.effective_gfactors_spin1 == []:
                resonance_frequencies_spin1, effective_gfactors_spin1 = spins[idx_spin1].res_freq(field_orientations_spin1, experiment.magnetic_field)
            else:
                effective_gfactors_spin1 = self.effective_gfactors_spin1
            resonance_frequencies_spin2, effective_gfactors_spin2 = spins[idx_spin2].res_freq(field_orientations_spin2, experiment.magnetic_field)
            timings.append(['Resonance frequencies', str(datetime.timedelta(seconds = time.time()-time_start))])
            time_start = time.time()
            # Detection probabilities
            if self.detection_probabilities_spin1 == {}:
                detection_probabilities_spin1 = experiment.detection_probability(resonance_frequencies_spin1, spins[idx_spin1].int_res_freq)
            else:
                detection_probabilities_spin1 = self.detection_probabilities_spin1[experiment.name]
            detection_probabilities_spin2 = experiment.detection_probability(resonance_frequencies_spin2, spins[idx_spin2].int_res_freq)
            # Pump probabilities
            if experiment.technique == 'peldor': 
                if self.pump_probabilities_spin1 == {}:
                    pump_probabilities_spin1 = np.where(detection_probabilities_spin2 > self.excitation_threshold,
                                                        experiment.pump_probability(resonance_frequencies_spin1, spins[idx_spin1].int_res_freq), 0.0)
                else:
                    pump_probabilities_spin1 = np.where(detection_probabilities_spin2 > self.excitation_threshold,
                                                        self.pump_probabilities_spin1[experiment.name], 0.0)
                pump_probabilities_spin2 = np.where(detection_probabilities_spin1 > self.excitation_threshold,
                                                    experiment.pump_probability(resonance_frequencies_spin2, spins[idx_spin2].int_res_freq), 0.0)   
            elif experiment.technique == 'ridme':
                if self.pump_probabilities_spin1 == {}:
                    pump_probabilities_spin1 = np.where(detection_probabilities_spin2 > self.excitation_threshold,
                                                        experiment.pump_probability(spins[idx_spin1].T1, spins[idx_spin1].g_anisotropy_in_dipolar_coupling, effective_gfactors_spin1), 0.0)
                else:
                    pump_probabilities_spin1 = np.where(detection_probabilities_spin2 > self.excitation_threshold,
                                                        self.pump_probabilities_spin1[experiment.name], 0.0)
                pump_probabilities_spin2 = np.where(detection_probabilities_spin1 > self.excitation_threshold,                                    
                                                    experiment.pump_probability(spins[idx_spin2].T1, spins[idx_spin2].g_anisotropy_in_dipolar_coupling, effective_gfactors_spin2), 0.0)   
            indices_nonzero_probabilities_spin1 = np.where(pump_probabilities_spin1 > self.excitation_threshold)[0]
            indices_nonzero_probabilities_spin2 = np.where(pump_probabilities_spin2 > self.excitation_threshold)[0]
            indices_nonzero_probabilities = np.unique(np.concatenate((indices_nonzero_probabilities_spin1, indices_nonzero_probabilities_spin2), axis=None))
            indices_nonzero_probabilities = np.sort(indices_nonzero_probabilities, axis=None)
            detection_probabilities_spin1 = detection_probabilities_spin1[indices_nonzero_probabilities]
            detection_probabilities_spin2 = detection_probabilities_spin2[indices_nonzero_probabilities]
            pump_probabilities_spin1 = pump_probabilities_spin1[indices_nonzero_probabilities]
            pump_probabilities_spin2 = pump_probabilities_spin2[indices_nonzero_probabilities]
            timings.append(['Detection/pump probabilities', str(datetime.timedelta(seconds = time.time()-time_start))])
            time_start = time.time()
            # Modulation depths
            amplitudes = detection_probabilities_spin1 + detection_probabilities_spin2
            modulation_amplitudes = detection_probabilities_spin1 * pump_probabilities_spin2 + \
                                    detection_probabilities_spin2 * pump_probabilities_spin1
            modulation_depths = modulation_amplitudes / np.sum(amplitudes)
            total_modulation_depth = np.sum(modulation_depths)
            timings.append(['Modulation depths', str(datetime.timedelta(seconds = time.time()-time_start))])
            time_start = time.time()
            # Modulation frequencies
            field_orientations = self.field_orientations[indices_nonzero_probabilities]
            r_values = r_values[indices_nonzero_probabilities]
            r_orientations = r_orientations[indices_nonzero_probabilities]
            effective_gfactors_spin1 = effective_gfactors_spin1[indices_nonzero_probabilities]
            effective_gfactors_spin2 = effective_gfactors_spin2[indices_nonzero_probabilities] 
            if not spins[idx_spin1].g_anisotropy_in_dipolar_coupling and not spins[idx_spin2].g_anisotropy_in_dipolar_coupling:
                angular_term = 1.0 - 3.0 * np.sum(r_orientations * field_orientations, axis=1)**2
            elif spins[idx_spin1].g_anisotropy_in_dipolar_coupling and not spins[idx_spin2].g_anisotropy_in_dipolar_coupling:
                quantization_axes_spin1 = spins[0].quantization_axis(field_orientations, effective_gfactors_spin1)
                angular_term = 1.0 - 3.0 * np.sum(r_orientations * quantization_axes_spin1, axis=1) * \
                                           np.sum(r_orientations * field_orientations, axis=1)
            elif not spins[idx_spin1].g_anisotropy_in_dipolar_coupling and spins[idx_spin2].g_anisotropy_in_dipolar_coupling:
                field_orientations_spin2 = field_orientations_spin2[indices_nonzero_probabilities]
                spin_frame_rotations_spin2 = spin_frame_rotations_spin2[indices_nonzero_probabilities]
                r_orientations_spin2 = rotate_coordinate_system(r_orientations, spin_frame_rotations_spin2, self.separate_grids)
                quantization_axes_spin2 = spins[idx_spin2].quantization_axis(field_orientations_spin2, effective_gfactors_spin2)
                angular_term = 1.0 - 3.0 * np.sum(r_orientations * field_orientations, axis=1) * \
                                           np.sum(r_orientations_spin2 * quantization_axes_spin2, axis=1) 
            elif spins[idx_spin1].g_anisotropy_in_dipolar_coupling and spins[idx_spin2].g_anisotropy_in_dipolar_coupling:
                quantization_axes_spin1 = spins[0].quantization_axis(field_orientations, effective_gfactors_spin1)
                field_orientations_spin2 = field_orientations_spin2[indices_nonzero_probabilities]
                spin_frame_rotations_spin2 = spin_frame_rotations_spin2[indices_nonzero_probabilities]
                r_orientations_spin2 = rotate_coordinate_system(r_orientations, spin_frame_rotations_spin2, self.separate_grids)
                quantization_axes_spin2 = spins[idx_spin2].quantization_axis(field_orientations_spin2, effective_gfactors_spin2)
                quantization_axes_spin2_ref = rotate_coordinate_system(quantization_axes_spin2, spin_frame_rotations_spin2.inv(), self.separate_grids)
                angular_term = np.sum(quantization_axes_spin1 * quantization_axes_spin2_ref, axis=1) - \
                                      3.0 * np.sum(r_orientations * quantization_axes_spin1, axis=1) * \
                                            np.sum(r_orientations_spin2 * quantization_axes_spin2, axis=1)
            dipolar_frequencies = const['Fdd'] * effective_gfactors_spin1 * effective_gfactors_spin2 * angular_term / r_values**3
            timings.append(['Dipolar frequencies', str(datetime.timedelta(seconds = time.time()-time_start))])
            time_start = time.time()
            # Time trace
            simulated_time_trace = self.time_trace_from_dipolar_spectrum(experiment, dipolar_frequencies, modulation_depths)
            timings.append(['PDS time trace', str(datetime.timedelta(seconds = time.time()-time_start))])
            # Messages
            if display_messages:
                print('Spins no. {0} and {1}'.format(idx_spin1, idx_spin2))
                print('Number of Monte-Carlo samples with non-zero weights: {0} out of {1}'.format(indices_nonzero_probabilities.size, self.mc_sample_size))
                for instance in timings:
                    print('\t {:<30} {:<30}'.format(instance[0], instance[1]))
            return simulated_time_trace, total_modulation_depth
        else:
            timings = [['Timings:', '']]
            time_start = time.time()
            # Random orientations of the applied static magnetic field in the reference frame
            if self.field_orientations == []:
                self.field_orientations = self.set_field_orientations()
            # Coordinates of spin 1 in the reference frame
            coordinates_spin1 = self.set_coordinates(variables['r_mean'][idx_spin1-1], variables['r_width'][idx_spin1-1], 
                                                     variables['xi_mean'][idx_spin1-1], variables['xi_width'][idx_spin1-1], 
                                                     variables['phi_mean'][idx_spin1-1], variables['phi_width'][idx_spin1-1], 
                                                     variables['rel_prob'][idx_spin1-1])
                                                     
            # Coordinates of spin 2 in the reference frame
            coordinates_spin2 = self.set_coordinates(variables['r_mean'][idx_spin2-1], variables['r_width'][idx_spin2-1], 
                                                     variables['xi_mean'][idx_spin2-1], variables['xi_width'][idx_spin2-1], 
                                                     variables['phi_mean'][idx_spin2-1], variables['phi_width'][idx_spin2-1], 
                                                     variables['rel_prob'][idx_spin2-1])
            r_vectors = coordinates_spin2 - coordinates_spin1
            # Distance values
            r_values = np.sqrt(np.sum(r_vectors**2, axis=1)) 
            # Orientations of the distance vector in the reference frame
            r_orientations = r_vectors / r_values.reshape(r_vectors.shape[0],1)
            # Rotation matrices transforming the reference frame into the spin frames
            spin_frame_rotations_spin1 = self.set_spin_frame_rotations(variables['alpha_mean'][idx_spin1-1], variables['alpha_width'][idx_spin1-1], 
                                                                       variables['beta_mean'][idx_spin1-1], variables['beta_width'][idx_spin1-1], 
                                                                       variables['gamma_mean'][idx_spin1-1], variables['gamma_width'][idx_spin1-1], 
                                                                       variables['rel_prob'][idx_spin1-1])
            spin_frame_rotations_spin2 = self.set_spin_frame_rotations(variables['alpha_mean'][idx_spin2-1], variables['alpha_width'][idx_spin2-1], 
                                                                       variables['beta_mean'][idx_spin2-1], variables['beta_width'][idx_spin2-1], 
                                                                       variables['gamma_mean'][idx_spin2-1], variables['gamma_width'][idx_spin2-1], 
                                                                       variables['rel_prob'][idx_spin2-1])                                                                                                 
            timings.append(['Monte-Carlo samples', str(datetime.timedelta(seconds = time.time()-time_start))])
            time_start = time.time()
            # Orientations of the applied static magnetic field in both spin frames
            field_orientations_spin1 = rotate_coordinate_system(self.field_orientations, spin_frame_rotations_spin1, self.separate_grids)
            field_orientations_spin2 = rotate_coordinate_system(self.field_orientations, spin_frame_rotations_spin2, self.separate_grids)
            # Resonance frequencies and/or effective g-values of both spins
            resonance_frequencies_spin1, effective_gfactors_spin1 = spins[idx_spin1].res_freq(field_orientations_spin1, experiment.magnetic_field)    
            resonance_frequencies_spin2, effective_gfactors_spin2 = spins[idx_spin2].res_freq(field_orientations_spin2, experiment.magnetic_field)
            timings.append(['Resonance frequencies', str(datetime.timedelta(seconds = time.time()-time_start))])
            time_start = time.time()
            # Detection probabilities
            detection_probabilities_spin1 = experiment.detection_probability(resonance_frequencies_spin1, spins[idx_spin1].int_res_freq)
            detection_probabilities_spin2 = experiment.detection_probability(resonance_frequencies_spin2, spins[idx_spin2].int_res_freq)
            # Pump probabilities
            if experiment.technique == 'peldor':
                pump_probabilities_spin1 = np.where(detection_probabilities_spin2 > self.excitation_threshold,
                                                    experiment.pump_probability(resonance_frequencies_spin1, spins[idx_spin1].int_res_freq), 0.0)
                pump_probabilities_spin2 = np.where(detection_probabilities_spin1 > self.excitation_threshold,
                                                    experiment.pump_probability(resonance_frequencies_spin2, spins[idx_spin2].int_res_freq), 0.0)   
            elif experiment.technique == 'ridme':
                pump_probabilities_spin1 = np.where(detection_probabilities_spin2 > self.excitation_threshold,
                                                    experiment.pump_probability(spins[idx_spin1].T1, spins[idx_spin1].g_anisotropy_in_dipolar_coupling, effective_gfactors_spin1), 0.0)
                pump_probabilities_spin2 = np.where(detection_probabilities_spin1 > self.excitation_threshold,                                    
                                                    experiment.pump_probability(spins[idx_spin2].T1, spins[idx_spin2].g_anisotropy_in_dipolar_coupling, effective_gfactors_spin2), 0.0) 
            indices_nonzero_probabilities_spin1 = np.where(pump_probabilities_spin1 > self.excitation_threshold)[0]
            indices_nonzero_probabilities_spin2 = np.where(pump_probabilities_spin2 > self.excitation_threshold)[0]
            indices_nonzero_probabilities = np.unique(np.concatenate((indices_nonzero_probabilities_spin1, indices_nonzero_probabilities_spin2), axis=None))
            indices_nonzero_probabilities = np.sort(indices_nonzero_probabilities, axis=None)
            detection_probabilities_spin1 = detection_probabilities_spin1[indices_nonzero_probabilities]
            detection_probabilities_spin2 = detection_probabilities_spin2[indices_nonzero_probabilities]
            pump_probabilities_spin1 = pump_probabilities_spin1[indices_nonzero_probabilities]
            pump_probabilities_spin2 = pump_probabilities_spin2[indices_nonzero_probabilities]
            timings.append(['Detection/pump probabilities', str(datetime.timedelta(seconds = time.time()-time_start))])
            time_start = time.time() 
            # Modulation depths
            amplitudes = detection_probabilities_spin1 + detection_probabilities_spin2
            modulation_amplitudes = detection_probabilities_spin1 * pump_probabilities_spin2 + \
                                    detection_probabilities_spin2 * pump_probabilities_spin1
            modulation_depths = modulation_amplitudes / np.sum(amplitudes)
            total_modulation_depth = np.sum(modulation_depths)
            timings.append(['Modulation depths', str(datetime.timedelta(seconds = time.time()-time_start))])
            time_start = time.time()
            # Modulation frequencies
            field_orientations = self.field_orientations[indices_nonzero_probabilities]
            r_values = r_values[indices_nonzero_probabilities]
            r_orientations = r_orientations[indices_nonzero_probabilities]
            effective_gfactors_spin1 = effective_gfactors_spin1[indices_nonzero_probabilities]
            effective_gfactors_spin2 = effective_gfactors_spin2[indices_nonzero_probabilities]
            if not spins[idx_spin1].g_anisotropy_in_dipolar_coupling and not spins[idx_spin2].g_anisotropy_in_dipolar_coupling:
                angular_term = 1.0 - 3.0 * np.sum(r_orientations * field_orientations, axis=1)**2
            elif spins[idx_spin1].g_anisotropy_in_dipolar_coupling and not spins[idx_spin2].g_anisotropy_in_dipolar_coupling:
                field_orientations_spin1 = field_orientations_spin1[indices_nonzero_probabilities]
                spin_frame_rotations_spin1 = spin_frame_rotations_spin1[indices_nonzero_probabilities]
                r_orientations_spin1 = rotate_coordinate_system(r_orientations, spin_frame_rotations_spin1, self.separate_grids)
                quantization_axes_spin1 = spins[idx_spin1].quantization_axis(field_orientations_spin1, effective_gfactors_spin1)
                angular_term = 1.0 - 3.0 * np.sum(r_orientations_spin1 * quantization_axes_spin1, axis=1) * \
                                           np.sum(r_orientations * field_orientations, axis=1)
            elif not spins[idx_spin1].g_anisotropy_in_dipolar_coupling and spins[idx_spin2].g_anisotropy_in_dipolar_coupling:
                field_orientations_spin2 = field_orientations_spin2[indices_nonzero_probabilities]
                spin_frame_rotations_spin2 = spin_frame_rotations_spin2[indices_nonzero_probabilities]
                r_orientations_spin2 = rotate_coordinate_system(r_orientations, spin_frame_rotations_spin2, self.separate_grids)
                quantization_axes_spin2 = spins[idx_spin2].quantization_axis(field_orientations_spin2, effective_gfactors_spin2)
                angular_term = 1.0 - 3.0 * np.sum(r_orientations * field_orientations, axis=1) * \
                                           np.sum(r_orientations_spin2 * quantization_axes_spin2, axis=1) 
            elif spins[idx_spin1].g_anisotropy_in_dipolar_coupling and spins[idx_spin2].g_anisotropy_in_dipolar_coupling:
                field_orientations_spin1 = field_orientations_spin1[indices_nonzero_probabilities]
                field_orientations_spin2 = field_orientations_spin2[indices_nonzero_probabilities]
                spin_frame_rotations_spin1 = spin_frame_rotations_spin1[indices_nonzero_probabilities]
                spin_frame_rotations_spin1 = spin_frame_rotations_spin2[indices_nonzero_probabilities]
                r_orientations_spin1 = rotate_coordinate_system(r_orientations, spin_frame_rotations_spin1, self.separate_grids)
                r_orientations_spin2 = rotate_coordinate_system(r_orientations, spin_frame_rotations_spin2, self.separate_grids)
                quantization_axes_spin1 = spins[idx_spin1].quantization_axis(field_orientations_spin1, effective_gfactors_spin1)
                quantization_axes_spin2 = spins[idx_spin2].quantization_axis(field_orientations_spin2, effective_gfactors_spin2)
                quantization_axes_spin1_ref = rotate_coordinate_system(quantization_axes_spin1, spin_frame_rotations_spin1, self.separate_grids)
                quantization_axes_spin2_ref = rotate_coordinate_system(quantization_axes_spin2, spin_frame_rotations_spin2, self.separate_grids)
                angular_term = np.sum(quantization_axes_spin1_ref * quantization_axes_spin2_ref, axis=1) - \
                                      3.0 * np.sum(r_orientations_spin1 * quantization_axes_spin1, axis=1) * \
                                            np.sum(r_orientations_spin2 * quantization_axes_spin2, axis=1)
            dipolar_frequencies = const['Fdd'] * effective_gfactors_spin1 * effective_gfactors_spin2 * angular_term / r_values**3
            timings.append(['Dipolar frequencies', str(datetime.timedelta(seconds = time.time()-time_start))])
            time_start = time.time()
            # Time trace
            simulated_time_trace = self.time_trace_from_dipolar_spectrum(experiment, dipolar_frequencies, modulation_depths)
            timings.append(['PDS time trace', str(datetime.timedelta(seconds = time.time()-time_start))])
            # Messages
            if display_messages:
                print('\t Spins no. {0} and {1}'.format(idx_spin1, idx_spin2))
                print('\t Number of Monte-Carlo samples with non-zero weights: {0} out of {1}'.format(indices_nonzero_probabilities.size, self.mc_sample_size))
                for instance in timings:
                    print('\t {:<30} {:<30}'.format(instance[0], instance[1]))
            return simulated_time_trace, total_modulation_depth
    
    def compute_time_trace(self, experiment, spins, variables, display_messages=True):
        ''' Computes a PDS time trace for a given set of variables '''
        if display_messages:
            print('\nComputing the time trace of the experiment \'{0}\'...'.format(experiment.name))
        num_spins = len(spins)
        if num_spins == 2:
            simulated_time_trace, modulation_depth = self.compute_time_trace_via_monte_carlo(experiment, spins, variables)
        else:
            simulated_time_trace = {}
            simulated_time_trace['t'] = experiment.t
            simulated_time_trace['s'] = np.ones(experiment.t.size)
            residual_amplitude = 1.0
            for i in range(num_spins-1):
                for j in range(i+1, num_spins):
                    if self.integration_method == 'monte_carlo':
                        two_spin_time_trace, two_spin_modulation_depth = self.compute_time_trace_via_monte_carlo_multispin(experiment, spins, variables, i, j)  
                    simulated_time_trace['s'] = simulated_time_trace['s'] * two_spin_time_trace['s']
                    residual_amplitude = residual_amplitude * (1.0 - two_spin_modulation_depth)
            modulation_depth = 1.0 - residual_amplitude
        # Rescale the modulation depth
        if self.fit_modulation_depth:
            simulated_time_trace['s'], modulation_depth_scale_factor = self.rescale_modulation_depth(simulated_time_trace['s'], modulation_depth, experiment.modulation_depth)
            if display_messages:
                print('Scale factor of the modulation depth: {0:<15.3}'.format(modulation_depth_scale_factor))
        else:
            modulation_depth_scale_factor = 1.0
        # Compute chi2
        if display_messages:
            chi2_value = chi2(simulated_time_trace['s'], experiment.s, experiment.noise_std)
            if self.fit_modulation_depth:
                degrees_of_freedom = experiment.s.size - len(variables) - 1
            else:
                degrees_of_freedom = experiment.s.size - len(variables)
            reduced_chi2_value = chi2_value / float(degrees_of_freedom)
            if experiment.noise_std:
                print('Chi2: {0:<15.3}   Reduced chi2: {1:<15.3}'.format((chi2_value, reduced_chi2_value)))
            else:
                print('Chi2 (noise std = 1): {0:<15.3}'.format(chi2_value))
        return simulated_time_trace, modulation_depth_scale_factor
    
    def compute_time_traces(self, experiments, spins, variables, display_messages=True):
        ''' Computes PDS time traces for a given set of variables '''
        simulated_time_traces = []
        modulation_depth_scale_factors = []
        for experiment in experiments:
            simulated_time_trace, modulation_depth_scale_factor = self.compute_time_trace(experiment, spins, variables, display_messages)
            simulated_time_traces.append(simulated_time_trace)
            modulation_depth_scale_factors.append(modulation_depth_scale_factor)
        return simulated_time_traces, modulation_depth_scale_factors
    
    def epr_spectrum(self, spins, field_value):
        ''' Computes an EPR spectrum of a spin system at a single magnetic field '''
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
    
    def epr_spectra(self, spins, experiments):
        ''' Computes an EPR spectrum of a spin system at multiple magnetic fields '''
        print('\nComputing the EPR spectrum of the spin system in the frequency domain...')
        epr_spectra = []
        for experiment in experiments:
            epr_spectrum = self.epr_spectrum(spins, experiment.magnetic_field)
            epr_spectra.append(epr_spectrum)
        return epr_spectra
    
    def bandwidths(self, experiments):
        ''' Computes the bandwidths of detection and pump pulses '''
        print('\nComputing the bandwidths of detection and pump pulses...')
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
    
    def precalculations(self, experiments, spins):
        ''' Pre-compute the detection and pump probabilities for spin 1'''
        print('\nPre-compute the detection and pump probabilities for spin 1...')
        if self.field_orientations == []:
            # Random orientations of the static magnetic field
            self.field_orientations = self.set_field_orientations()
        # Orientations of the applied static magnetic field in the frame of spin 1
        field_orientations_spin1 = self.field_orientations 
        if self.detection_probabilities_spin1 != {}:
            for experiment in experiments:
                # Resonance frequencies and effective g-values of spin 1
                resonance_frequencies_spin1, self.effective_gfactors_spin1 = spins[0].res_freq(field_orientations_spin1, experiment.magnetic_field)
                # Detection probabilities
                self.detection_probabilities_spin1[experiment.name] = experiment.detection_probability(resonance_frequencies_spin1, spins[0].int_res_freq)
                # Pump probabilities
                if experiment.technique == 'peldor': 
                    self.pump_probabilities_spin1[experiment.name] = experiment.pump_probability(resonance_frequencies_spin1, spins[0].int_res_freq)    
                elif experiment.technique == 'ridme':
                    self.pump_probabilities_spin1[experiment.name] = experiment.pump_probability(spins[0].T1, spins[0].g_anisotropy_in_dipolar_coupling, self.effective_gfactors_spin1)  