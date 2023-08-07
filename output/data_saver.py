import os
import numpy as np
import errno
import datetime
import shutil
from output.simulation.save_epr_spectrum import save_epr_spectrum
from output.simulation.save_bandwidth import save_bandwidth
from output.simulation.save_background_time_trace import save_background_time_trace
from output.simulation.save_background_free_time_trace import save_background_free_time_trace
from output.simulation.save_simulated_spectrum import save_simulated_spectrum
from output.simulation.save_dipolar_angle_distribution  import save_dipolar_angle_distribution
from output.simulation.save_simulated_time_trace import save_simulated_time_trace
from output.fitting.save_score import save_score, save_score_all_runs
from output.fitting.save_model_parameters import save_model_parameters, save_model_parameters_multiple_runs
from output.fitting.save_background_parameters import save_background_parameters
from output.fitting.save_symmetry_related_solutions import save_symmetry_related_solutions
from output.error_analysis.save_error_surface import save_error_surface
from output.error_analysis.save_error_profile import save_error_profile

class DataSaver:
    ''' Saves the output data of the program '''
    
    def __init__(self, save_data, save_figures):
        self.save_data = save_data
        self.save_figures = save_figures
        self.directory = ''  
    
    def create_output_directory(self, parent_directory, filepath_config):
        ''' Creates an output directory ''' 
        if self.save_data or self.save_figures:
            config_directory, config_name = os.path.split(os.path.abspath(filepath_config))
            if parent_directory != '':
                output_directory = parent_directory
            else:
                output_directory = config_directory
                
            now = datetime.datetime.now()
            folder = now.strftime('%Y-%m-%d_%H-%M') + '_' + config_name[:-4]
            output_directory = output_directory + '/' + folder + '/'
            try:
                os.makedirs(output_directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise      
            try:
                shutil.copy2(filepath_config, output_directory + config_name)
            except: 
                pass
            self.directory = output_directory
   
    def save_simulation_output(self, epr_spectra, bandwidths, simulated_time_traces, background_time_traces, background_free_time_traces, 
                               simulated_spectra, dipolar_angle_distributions, background_parameters, background, experiments):
        ''' Saves the simulation output '''
        self.save_bandwidths(epr_spectra, bandwidths, experiments)
        self.save_simulated_time_traces(simulated_time_traces, [], experiments)   
        self.save_background_time_traces(background_time_traces, [], experiments)
        self.save_background_free_time_traces(background_free_time_traces, experiments)
        self.save_simulated_spectra(simulated_spectra, experiments)
        self.save_dipolar_angle_distributions(dipolar_angle_distributions, experiments)
        self.save_background_parameters(background_parameters, [], background, experiments)
        
    def save_fitting_output(self, epr_spectra, bandwidths, idx_best_solution, score_all_runs, optimized_model_parameters_all_runs, 
                            optimized_background_parameters, symmetry_related_solutions, simulated_time_traces, background_time_traces, 
                            background_free_time_traces, simulated_spectra, dipolar_angle_distributions, fitting_parameters, experiments, background):
        ''' Saves the fitting output '''
        self.save_bandwidths(epr_spectra, bandwidths, experiments)
        self.save_score(score_all_runs[idx_best_solution])
        self.save_score_multiple_runs(score_all_runs)
        self.save_model_parameters(optimized_model_parameters_all_runs[idx_best_solution], [], fitting_parameters)
        self.save_model_parameters_multiple_runs(optimized_model_parameters_all_runs, fitting_parameters)
        self.save_symmetry_related_solutions(symmetry_related_solutions, fitting_parameters)
        self.save_background_parameters(optimized_background_parameters, [], background, experiments)
        self.save_fits(simulated_time_traces, [], experiments)
        self.save_background_time_traces(background_time_traces, [], experiments)
        self.save_background_free_time_traces(background_free_time_traces, experiments)
        self.save_simulated_spectra(simulated_spectra, experiments)
        self.save_dipolar_angle_distributions(dipolar_angle_distributions, experiments)

    def save_error_analysis_output(self, optimized_model_parameters, model_parameter_errors,
                                   optimized_background_parameters, background_parameter_errors,
                                   error_surfaces, error_profiles,
                                   background, fitting_parameters, error_analysis_parameters, experiments,
                                   simulated_time_traces, error_bars_simulated_time_traces,
                                   background_time_traces, error_bars_background_time_traces,
                                   background_free_time_traces, simulated_spectra, dipolar_angle_distributions):    
        ''' Saves the error analysis output '''
        self.save_model_parameters(optimized_model_parameters, model_parameter_errors, fitting_parameters)
        self.save_background_parameters(optimized_background_parameters, background_parameter_errors, background, experiments)
        self.save_error_surfaces(error_surfaces, error_analysis_parameters)
        self.save_error_profiles(error_profiles, error_analysis_parameters)
        self.save_fits(simulated_time_traces, error_bars_simulated_time_traces, experiments)
        self.save_background_time_traces(background_time_traces, error_bars_background_time_traces, experiments)
        self.save_background_free_time_traces(background_free_time_traces, experiments)
        self.save_simulated_spectra(simulated_spectra, experiments)
        self.save_dipolar_angle_distributions(dipolar_angle_distributions, experiments)

    def save_epr_spectrum(self, spectrum, experiment_name):
        ''' Saves a simulated EPR spectrum '''
        if self.save_data:
            filepath = self.directory + 'epr_spectrum_' + experiment_name + '.dat'
            save_epr_spectrum(spectrum, filepath)

    def save_bandwidths(self, epr_spectra, bandwidths, experiments):
        ''' Saves the bandwidths of detection and pump pulses'''
        if self.save_data:
            for i in range(len(experiments)):
                filepath = self.directory + 'epr_spectrum_' + experiments[i].name + '.dat'
                save_epr_spectrum(epr_spectra[i], filepath)
                for key in bandwidths[i]:
                    filepath = self.directory + key + '_' + experiments[i].name + '.dat'
                    save_bandwidth(bandwidths[i][key], filepath)

    def save_background_time_traces(self, background_time_traces, error_bars_background_time_traces, experiments):
        ''' Saves the background parts of PDS time traces '''
        if self.save_data:
            for i in range(len(experiments)):
                filepath = self.directory + 'background_' + experiments[i].name + '.dat'
                if error_bars_background_time_traces != []:
                    save_background_time_trace(background_time_traces[i], error_bars_background_time_traces[i], experiments[i], filepath)
                else:
                    save_background_time_trace(background_time_traces[i], [], experiments[i], filepath)
                    
    def save_background_free_time_traces(self, background_free_time_traces, experiments):
        ''' Saves the background-free parts of PDS time traces '''
        if self.save_data:
            for i in range(len(experiments)):
                filepath = self.directory + 'background_free_time_trace_' + experiments[i].name + '.dat'
                save_background_free_time_trace(background_free_time_traces[i], filepath)
    
    def save_simulated_spectra(self, simulated_spectra, experiments):
        ''' Saves simulated PDS spectra '''
        if self.save_data:
            for i in range(len(experiments)):
                filepath = self.directory + 'dipolar_spectrum_' + experiments[i].name + '.dat'
                save_simulated_spectrum(simulated_spectra[i], filepath)
    
    def save_dipolar_angle_distributions(self, dipolar_angle_distributions, experiments):
        ''' Saves simulated distributions of dipolar angles '''
        if self.save_data:
            for i in range(len(experiments)):
                filepath = self.directory + 'dipolar_angles_' + experiments[i].name + '.dat'
                save_dipolar_angle_distribution(dipolar_angle_distributions[i], filepath)
                
    def save_simulated_time_traces(self, simulated_time_traces, error_bars_simulated_time_traces, experiments):
        ''' Saves simulated PDS time traces '''
        if self.save_data:
            for i in range(len(experiments)):
                filepath = self.directory + 'time_trace_' + experiments[i].name + '.dat'
                if error_bars_simulated_time_traces != []:
                    save_simulated_time_trace(simulated_time_traces[i], error_bars_simulated_time_traces[i], experiments[i], filepath)
                else:
                    save_simulated_time_trace(simulated_time_traces[i], [], experiments[i], filepath)
    
    def save_score(self, score):
        ''' Saves the score as a function of optimization step '''
        if self.save_data:
            filepath = self.directory + 'score.dat'
            save_score(score, filepath)
    
    def save_score_multiple_runs(self, score_multiple_runs):
        ''' Saves the score as a function of optimization step for multiple optimization runs '''
        if self.save_data:
            if len(score_multiple_runs) > 1:
                for i in range(len(score_multiple_runs)):
                    filepath = self.directory + 'score_run' + str(i+1) + '.dat'
                    save_score(score_multiple_runs[i], filepath)
                filepath = self.directory + 'score_all_runs.dat'
                save_score_all_runs(score_multiple_runs, filepath)
                    
    def save_model_parameters(self, optimized_model_parameters, model_parameter_errors, fitting_parameters):    
        ''' Saves the model parameters ''' 
        if self.save_data:
            filepath = self.directory + 'fitting_parameters.dat'
            save_model_parameters(optimized_model_parameters, model_parameter_errors, fitting_parameters, filepath)
    
    def save_model_parameters_multiple_runs(self, optimized_model_parameters_all_runs, fitting_parameters):    
        ''' Saves the model parameters obtained in multiple optimizations ''' 
        if self.save_data:
            if len(optimized_model_parameters_all_runs) > 1:
                for i in range(len(optimized_model_parameters_all_runs)):
                    filepath = self.directory + 'fitting_parameters_run' + str(i+1) + '.dat'
                    save_model_parameters(optimized_model_parameters_all_runs[i], [], fitting_parameters, filepath)
                filepath = self.directory + 'fitting_parameters_all_runs.dat'
                save_model_parameters_multiple_runs(optimized_model_parameters_all_runs, fitting_parameters, filepath)
    
    def save_symmetry_related_solutions(self, symmetry_related_solutions, fitting_parameters):
        ''' Saves the symmetry-related sets of model parameters ''' 
        if self.save_data:
            filepath = self.directory + 'symmetry_related_solutions.dat'
            save_symmetry_related_solutions(symmetry_related_solutions, fitting_parameters, filepath)
     
    def save_background_parameters(self, optimized_background_parameters, background_parameter_errors, background, experiments):
        ''' Saves the background parameters of PDS time traces '''
        if self.save_data:
            filepath = self.directory + 'background_parameters.dat'
            save_background_parameters(optimized_background_parameters, background_parameter_errors, background, experiments, filepath)
      
    def save_fits(self, simulated_time_traces, error_bars_simulated_time_traces, experiments):
        ''' Saves the fit to PDS time traces '''
        if self.save_data:
            for i in range(len(experiments)):
                filepath = self.directory + 'fit_' + experiments[i].name + '.dat'
                if error_bars_simulated_time_traces != []:
                    save_simulated_time_trace(simulated_time_traces[i], error_bars_simulated_time_traces[i], experiments[i], filepath)
                else:
                    save_simulated_time_trace(simulated_time_traces[i], [], experiments[i], filepath)
    
    def save_error_surfaces(self, error_surfaces, error_analysis_parameters):
        ''' Saves error surfaces '''
        if self.save_data:
            for i in range(len(error_analysis_parameters)):
                filepath = self.directory + 'error_surface_' + str(i+1) + '.dat'
                save_error_surface(error_surfaces[i], error_analysis_parameters[i], filepath)
        
    def save_error_profiles(self, error_profiles, error_analysis_parameters):
        ''' Saves confidence intervals '''
        if self.save_data:
            c = 0
            for i in range(len(error_analysis_parameters)):
                for j in range(len(error_analysis_parameters[i])):
                    filepath = self.directory + 'error_profile_' + str(c+1) + '.dat'
                    save_error_profile(error_profiles[c], error_analysis_parameters[i][j], filepath)    
                    c += 1      