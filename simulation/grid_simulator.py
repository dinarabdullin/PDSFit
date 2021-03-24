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


class GridSimulator():
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
    
    def set_field_orientations_grid(self):
        ''' Powder-averging grid via Lebedev angular quadrature '''
        # in progress

    def set_r_values_grid(self, r_mean, r_width, rel_prob):
        ''' r-grid via Gauss-Legendre quadrature '''
        # in progress
        
    def set_r_orientations_grid(self, xi_mean, xi_width, phi_mean, phi_width, rel_prob):
        ''' 
        xi/phi-grid via Lebedev angular quadrature.
        It is  used to compute the orientations of the distance vector in the reference frame
        '''
        # in progress
        
    def set_spin_frame_rotations_grid(self, alpha_mean, alpha_width, beta_mean, beta_width, gamma_mean, gamma_width, rel_prob):
        '''
        alpha/beta/gamma-grid via Mitchell grid.
        It is used to compute rotation matrices transforming the reference frame into the spin frame
        '''
        # in progress
        
    def set_j_values_grid(self, j_mean, j_width, rel_prob):
        ''' j-grid via Gauss-Legendre quadrature '''
        # in progress
        
    def set_coordinates(self, r_mean, r_width, xi_mean, xi_width, phi_mean, phi_width, rel_prob):
        ''' 
        The values of r, xi, and phi from corresponding distributions P(r), P(xi), and P(phi)
        are used to compute the coordinates of the distance vector in the reference frame
        '''
        # in progress

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
        # in progress
        
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