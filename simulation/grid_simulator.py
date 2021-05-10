from numpy.lib.function_base import digitize
from GridIntegration.GridIntegration import GridIntegration
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


# added

from GridIntegration.Lebedev import LebedevAngularIntegration
from GridIntegration.Gauss_Legendre import Gauss_Legendre
from GridIntegration.Mitchell import Mitchell_Integration
from mathematics.distributions import normal_distribution, uniform_distribution, vonmises_distribution


class GridSimulator(Simulator):
    ''' Monte-Carlo Simulation class '''
    
    def __init__(self, calculation_settings):
        super().__init__(calculation_settings)
        self.grid_size = calculation_settings['grid_size']
        self.separate_grids = True
        self.frequency_increment_epr_spectrum = 0.001 # in GHz
        self.field_orientations = []
        self.effective_gfactors_spin1 = []
        self.detection_probabilities_spin1 = {}
        self.pump_probabilities_spin1 = {}

        self.lebedev = LebedevAngularIntegration()
        self.gauss_leg = Gauss_Legendre()
        self.mitchell = Mitchell_Integration()

        # this should later be copied into read config
        self.grid_functions = {'xi_field':lambda xi, phi: np.sin(xi)}
        for variable in ["r", "xi", "phi", "alpha", "beta", "gamma", "j"]:
            if calculation_settings['distributions'][variable] == "normal":
                self.grid_functions[variable] = normal_distribution
            elif  calculation_settings['distributions'][variable] == "vonmises":
                 self.grid_functions[variable] = vonmises_distribution
            elif calculation_settings['distributions'][variable] == "uniform":
                self.grid_functions[variable] = uniform_distribution

    
    def set_field_orientations_grid(self, treshold, for_spectrum = False):
        ''' Powder-averging grid via Lebedev angular quadrature '''
        # orientations of the field vector in cartesian coordinates (points on a unit sphere)
        if for_spectrum:
            # would have to be edited in read_config
            #grid_size = self.grid_size["powder_averaging_for_spectrum"]
            grid_size = 5810
        else: 
            grid_size = self.grid_size["powder_averaging"]
        field_orientations = self.lebedev.get_points(grid_size)
        field_weights = self.lebedev.get_weighted_summands((lambda xi, phi: np.sin(xi)), grid_size)
        field_orientations = field_orientations[field_weights>treshold] 
        field_weights = field_weights[field_weights>treshold]
        return field_orientations, field_weights


    def set_r_values_grid(self, r_mean, r_width, r_min, r_max, treshold):
        ''' r-grid via Gauss-Legendre quadrature '''
        r_values = self.gauss_leg.get_points(self.grid_size["distances"], r_min, r_max)
        function = lambda r: self.grid_functions['r'](r, {'mean':r_mean, 'width':r_width})
        r_values_weights = self.gauss_leg.get_weighted_summands(function, self.grid_size["distances"], r_min, r_max)
        r_values = r_values[r_values_weights>treshold]
        r_values_weights = r_values_weights[r_values_weights>treshold]
        return r_values, r_values_weights
        
    def set_r_orientations_grid(self, xi_mean, xi_width, phi_mean, phi_width, treshold):
        ''' 
        xi/phi-grid via Lebedev angular quadrature. (M)
        It is  used to compute the orientations of the distance vector in the reference frame
        '''
        # unit vector of the r_orientation in cartesian coordinates
        r_orientations = self.lebedev.get_points(self.grid_size['spherical_angles'])
        function = lambda xi, phi : (np.sin(xi)*self.grid_functions['xi'](xi,{'mean':xi_mean, 'width':xi_width})
                                    *self.grid_functions['phi'](phi, {'mean':phi_mean,'width':phi_width}))
        r_orientations_weights = self.lebedev.get_weighted_summands(function, self.grid_size['spherical_angles'])
        r_orientations = r_orientations[r_orientations_weights>treshold]
        r_orientations_weights = r_orientations_weights[r_orientations_weights>treshold]
        return r_orientations, r_orientations_weights
        
    def set_spin_frame_rotations_grid(self, alpha_mean, alpha_width, beta_mean, beta_width, gamma_mean, gamma_width, treshold):
        '''
        alpha/beta/gamma-grid via Mitchell grid.
        It is used to compute rotation matrices transforming the reference frame into the spin frame
        '''
        alpha_beta_gamma = self.mitchell.get_points(self.grid_size['rotations'])
        function = (lambda alpha, beta, gamma: np.sin(beta)*self.grid_functions['alpha'](alpha, {'mean':alpha_mean, 'width':alpha_width})*
                                            self.grid_functions['beta'](beta, {'mean':beta_mean,'width': beta_width})* 
                                            self.grid_functions['gamma'](gamma, {'mean':gamma_mean, 'width':gamma_width}))
        alpha_beta_gamma_weights = self.mitchell.get_weighted_summands(function, self.grid_size['rotations'])
        alpha_beta_gamma = alpha_beta_gamma[alpha_beta_gamma_weights>treshold]
        alpha_beta_gamma_weights = alpha_beta_gamma_weights[alpha_beta_gamma_weights>treshold]
        # take convention from calculation settings later
        spin_frame_rotations = Rotation.from_euler(self.euler_angles_convention, alpha_beta_gamma)
        return spin_frame_rotations, alpha_beta_gamma_weights
        
    def set_j_values_grid(self, j_mean, j_width, j_min, j_max, treshold):
        ''' j-grid via Gauss-Legendre quadrature '''
        #this needs to be in read config
        self.grid_size["j"] = 20
        j_values = self.gauss_leg.get_points(self.grid_size["j"], j_min, j_max)
        function = lambda r: self.grid_functions['r'](r, {'mean':j_mean, 'width':j_width})
        j_values_weights = self.gauss_leg.get_weighted_summands(function, self.grid_size["j"], j_min, j_max)
        j_values = j_values[j_values_weights>treshold]
        j_values_weights = j_values_weights[j_values_weights>treshold]
        return j_values, j_values_weights
        
    def set_coordinates(self, r_mean, r_width, r_min, r_max, xi_mean, xi_width, phi_mean, phi_width, treshold):
        ''' 
        The values of r, xi, and phi from corresponding distributions P(r), P(xi), and P(phi)
        are used to compute the coordinates of the distance vector in the reference frame
        '''
        # PROBLEM: different dimensions due to different grid sizes..
        r_values = self.set_r_values_grid(r_mean, r_width, r_max, r_min, treshold)
        r_orientation_cartesian = self.set_r_orientations_grid(xi_mean, xi_width, phi_mean, phi_width, treshold)
        ## scale orientation_unit_vector with it's corresponding length 
        #coordinates = r_orientation_cartesian * r_values.reshape(r_values.size, 1)
        #return coordinates

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
 
    def compute_time_trace_via_grids(self, experiment, spins, variables, idx_spin1=0, idx_spin2=1, display_messages=False):
        ''' Computes a PDS time trace via integration grids '''
        # this code would fit better into read config
        variables['r_min'] = [variables['r_mean'][0][0]-4*variables['r_width'][0][0]]
        variables['r_max'] = [variables['r_mean'][0][0]+4*variables['r_width'][0][0]]
        variables['j_min'] = [0]
        variables['j_max'] = [0]
        # in progress
        if self.field_orientations == []:
            self.field_orientations, self.field_orientations_weights = self.set_field_orientations_grid(treshold = 0.1)
        L = self.field_orientations.shape[0]
        r_values, r_values_weights = self.set_r_values_grid(variables['r_mean'][0], variables['r_width'][0], variables['r_min'][0], variables['r_max'][0], treshold=1e-4 )
        N = r_values.shape[0]
        r_orientations, r_orientations_weights = self.set_r_orientations_grid(variables['xi_mean'][0], variables['xi_width'][0], 
                                                 variables['phi_mean'][0], variables['phi_width'][0], 
                                                 treshold = 1e-7)
        M = r_orientations.shape[0]
        # K' rotation matrices                                         
        spin_frame_rotations_spin2, spin_frame_rotations_spin2_weights  = self.set_spin_frame_rotations_grid(variables['alpha_mean'][0], variables['alpha_width'][0], 
                                                                   variables['beta_mean'][0], variables['beta_width'][0], 
                                                                   variables['gamma_mean'][0], variables['gamma_width'][0], 
                                                                   treshold = 1e-6)
        K = spin_frame_rotations_spin2.__len__()
        j_values, j_values_weights = self.set_j_values_grid(variables['j_mean'][0], variables['j_width'][0], variables['j_min'][0], variables['j_max'][0], treshold=0)
        O = j_values.shape[0]
        #  field_orientations_spin1 shape: (L',3)
        field_orientations_spin1 = self.field_orientations
        # field_orientations_spin2 shape : ((L'*K'), 3)
        field_orientations_spin2 = rotate_coordinate_system(self.field_orientations, spin_frame_rotations_spin2, self.separate_grids)
        if self.effective_gfactors_spin1 == []:
            resonance_frequencies_spin1, effective_gfactors_spin1 = spins[0].res_freq(field_orientations_spin1, experiment.magnetic_field)
        else:
            effective_gfactors_spin1 = self.effective_gfactors_spin1
        # resonance_frequencies_spin2, effective_gfactors_spin2 shapes: (L'*K',3) , (L'*K',)
        resonance_frequencies_spin2, effective_gfactors_spin2 = spins[1].res_freq(field_orientations_spin2, experiment.magnetic_field)
        # detection_probabilities_spin1 shape : (L',)
        if self.detection_probabilities_spin1 == {}:
            detection_probabilities_spin1 = experiment.detection_probability(resonance_frequencies_spin1, spins[0].int_res_freq)
        else:
            detection_probabilities_spin1 = self.detection_probabilities_spin1[experiment.name]
        detection_probabilities_spin1_LK = np.repeat(detection_probabilities_spin1, K)
        # detection_probabilities_spin2_LK shape: (L'*K',)
        detection_probabilities_spin2_LK = experiment.detection_probability(resonance_frequencies_spin2, spins[1].int_res_freq)
        if experiment.technique == 'peldor': 
            if self.pump_probabilities_spin1 == {}:
                pump1_LK = np.repeat(experiment.pump_probability(resonance_frequencies_spin1, spins[0].int_res_freq), K)
                pump_probabilities_spin1_LK = np.where(detection_probabilities_spin2_LK > self.excitation_threshold, pump1_LK, 0.0)
                # pump_probabilities_spin1_LK now has shape (L'*K',)
            else:
                pump_probabilities_spin1_LK = np.where(detection_probabilities_spin2_LK > self.excitation_threshold,
                                                    self.pump_probabilities_spin1[experiment.name], 0.0)
            
            
            pump_probabilities_spin2_LK = np.where( detection_probabilities_spin1_LK> self.excitation_threshold,
                                                experiment.pump_probability(resonance_frequencies_spin2, spins[1].int_res_freq), 0.0) 
        print(detection_probabilities_spin1_LK.shape)
        # difficult to implement this. When removing zero probabilities, how can one say which dimension was shrinked and by how much?
        # but removing this code block makes previous np.where code obsolete..
        """ indices_nonzero_probabilities_spin1 = np.where(pump_probabilities_spin1_LK > self.excitation_threshold)[0]
        indices_nonzero_probabilities_spin2 = np.where(pump_probabilities_spin2_LK > self.excitation_threshold)[0]
        indices_nonzero_probabilities = np.unique(np.concatenate((indices_nonzero_probabilities_spin1, indices_nonzero_probabilities_spin2), axis=None))
        indices_nonzero_probabilities = np.sort(indices_nonzero_probabilities, axis=None)
        detection_probabilities_spin1_LK = detection_probabilities_spin1_LK[indices_nonzero_probabilities]
        detection_probabilities_spin2_LK = detection_probabilities_spin2_LK[indices_nonzero_probabilities]
        pump_probabilities_spin1_LK = pump_probabilities_spin1_LK[indices_nonzero_probabilities]
        pump_probabilities_spin2_LK = pump_probabilities_spin2_LK[indices_nonzero_probabilities] """
        
        amplitudes_LK = detection_probabilities_spin1_LK + detection_probabilities_spin2_LK
        modulation_amplitudes_LK = detection_probabilities_spin1_LK * pump_probabilities_spin2_LK + \
                                detection_probabilities_spin2_LK * pump_probabilities_spin1_LK
        modulation_depths_LK = modulation_amplitudes_LK / np.sum(amplitudes_LK)
        total_modulation_depth = np.sum(modulation_depths_LK)  #equals 0.23
        angular_term_LM = 1.0 - 3.0 * np.sum(np.repeat(r_orientations, L, axis = 0) * np.repeat(self.field_orientations, M, axis=0), axis=1)**2
        angular_term_LM = angular_term_LM.reshape(L, M)
        # angular_term_LM now has shape (L,M)
        # now calculation of dipolar_freqencies:
        # first multiply effgspin1 with effgspin2 to create array with shape (L,K)
        effective_g_product = effective_gfactors_spin1.reshape(L,1)* effective_gfactors_spin2.reshape(L,K)
        # extend effective_g_product (L,K) and angular_term (L,M) to (L,M,K) arrays to enable their multiplication
        effective_g_product = effective_g_product.repeat(M, axis=0).reshape(L,M,K)
        angular_term_tmp = angular_term_LM.repeat(K).reshape(L,M,K)
        # multiply eff_gp_roduct and angular_term_tmp, extend result (L,M,K) to (L,M,K,N), then divide by r_values (N,)
        dipolar_frequencies =  const['Fdd']*(angular_term_tmp*effective_g_product).repeat(N).reshape(L,M,K,N) * 1/r_values**3
        # dipolar_frequencies has shape (L,M,K,N)
        # neglect j_values for now (mean and width are 0 in the config file anyway)
        modulation_frequencies = dipolar_frequencies
        modulation_depths_LMKN = modulation_depths_LK.reshape(L,K).repeat(M, axis=0).repeat(N).reshape(L,M,K,N)
        simulated_time_trace = self.time_trace_from_dipolar_spectrum(experiment, modulation_frequencies, modulation_depths_LMKN)
        
        
        return simulated_time_trace, total_modulation_depth


    def compute_time_trace_via_grids_multispin(experiment, spins, variables, idx_spin1=0, idx_spin2=1, display_messages=False):
        ''' Computes a PDS time trace via integration grids (multi-spin version) '''
        # in progress
        
    def compute_time_trace(self, experiment, spins, variables, display_messages=True):
        ''' Computes a PDS time trace for a given set of variables '''
        if display_messages:
            print('\nComputing the time trace of the experiment \'{0}\'...'.format(experiment.name))
        num_spins = len(spins)
        if num_spins == 2:
            simulated_time_trace, modulation_depth = self.compute_time_trace_via_grids(experiment, spins, variables)
        else:
            simulated_time_trace = {}
            simulated_time_trace['t'] = experiment.t
            simulated_time_trace['s'] = np.ones(experiment.t.size)
            residual_amplitude = 1.0
            for i in range(num_spins-1):
                for j in range(i+1, num_spins):
                    two_spin_time_trace, _ = self.compute_time_trace_via_grids_multispin(experiment, spins, variables, i, j)    
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
        # in progress
        ''' Computes an EPR spectrum of a spin system at a single magnetic field '''
        # Random orientations of the static magnetic field
        self.field_orientations_spc, self.field_orientations_weights_spc = self.set_field_orientations_grid(0, for_spectrum = True)  
        # Resonance frequencies and their probabilities
        all_frequencies = []
        all_probabilities = []
        for spin in spins:
            # Resonance frequencies
            resonance_frequencies, effective_gvalues = spin.res_freq(self.field_orientations_spc, field_value)
            num_field_orientations = self.field_orientations_spc.shape[0]
            #weights = np.tile(spin.int_res_freq, (num_field_orientations,1))
            weights = spin.int_res_freq * self.field_orientations_weights_spc.reshape(num_field_orientations, 1)
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
        # in progress