import os
import errno
import datetime
import shutil
from output.simulation.save_epr_spectrum import save_epr_spectrum
from output.simulation.save_bandwidth import save_bandwidth
from output.simulation.save_simulated_time_trace import save_simulated_time_trace
from output.fitting.save_score import save_score
from output.fitting.save_fitting_parameters import save_fitting_parameters
from output.fitting.save_symmetry_related_solutions import save_symmetry_related_solutions
from output.error_analysis.save_score_vs_parameters import save_score_vs_parameters
from output.error_analysis.save_score_vs_parameter import save_score_vs_parameter

class DataSaver:
    ''' The class to save the program's output data '''
    
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
            folder = now.strftime('%Y-%m-%d_%H-%M')
            output_directory = output_directory + '/' + folder + '/'
            try:
                os.makedirs(output_directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise      
            shutil.copy2(filepath_config, output_directory + config_name)
            self.directory = output_directory
   
    def save_simulation_output(self, epr_spectra, bandwidths, simulated_time_traces, experiments):
        ''' Saves the simulation output '''
        # self.save_epr_spectrum(epr_spectra[0], experiments[0].name)
        self.save_bandwidths(epr_spectra, bandwidths, experiments)
        self.save_simulated_time_traces(simulated_time_traces, experiments)   
    
    def save_fitting_output(self, score, optimized_parameters, parameter_errors, symmetry_related_solutions, simulated_time_traces, fitting_parameters, experiments):
        ''' Saves the fitting output '''
        self.save_score(score)
        self.save_fitting_parameters(fitting_parameters['indices'], optimized_parameters, fitting_parameters['values'], parameter_errors)
        self.save_symmetry_related_solutions(symmetry_related_solutions, fitting_parameters['indices'])
        self.save_fits(simulated_time_traces, experiments)

    def save_error_analysis_output(self, optimized_parameters, parameter_errors, fitting_parameters, score_vs_parameter_subsets, score_vs_parameters, error_analysis_parameters):    
        ''' Saves the error analysis output '''
        self.save_fitting_parameters(fitting_parameters['indices'], optimized_parameters, fitting_parameters['values'], parameter_errors)
        self.save_error_surfaces(score_vs_parameter_subsets, error_analysis_parameters)
        self.save_confidence_intervals(score_vs_parameters, error_analysis_parameters)

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

    def save_simulated_time_traces(self, simulated_time_traces, experiments):
        ''' Saves simulated PDS time traces '''
        if self.save_data:
            for i in range(len(experiments)):
                filepath = self.directory + 'time_trace_' + experiments[i].name + '.dat'
                save_simulated_time_trace(simulated_time_traces[i], experiments[i].s, experiments[i].s_im, filepath)
    
    def save_score(self, score):
        ''' Saves the score as a function of optimization step '''
        if self.save_data:
            filepath = self.directory + 'score.dat'
            save_score(score, filepath)

    def save_fitting_parameters(self, parameter_indices, optimized_parameters, fixed_parameters, parameter_errors):    
        ''' Saves optimized and fixed fitting parameters ''' 
        if self.save_data:
            filepath = self.directory + 'fitting_parameters.dat'
            save_fitting_parameters(parameter_indices, optimized_parameters, fixed_parameters, parameter_errors, filepath)
    
    def save_symmetry_related_solutions(self, symmetry_related_solutions, parameters_indices):
        ''' Saves symmetry-related sets of fitting parameters ''' 
        if self.save_data:
            filepath = self.directory + 'symmetry_related_solutions.dat'
            save_symmetry_related_solutions(symmetry_related_solutions, parameters_indices, filepath)

    def save_fits(self, simulated_time_traces, experiments):
        ''' Saves fits to experimental PDS time traces '''
        if self.save_data:
            for i in range(len(experiments)):
                filepath = self.directory + 'fit_' + experiments[i].name + '.dat'
                save_simulated_time_trace(simulated_time_traces[i], experiments[i].s, experiments[i].s_im, filepath)
    
    def save_error_surfaces(self, score_vs_parameter_subsets, error_analysis_parameters):
        ''' Saves chi2 as a function of fitting parameters' subsets '''
        if self.save_data:
            for i in range(len(error_analysis_parameters)):
                filepath = self.directory + 'score_vs_parameters_' + str(i+1) + '.dat'
                save_score_vs_parameters(score_vs_parameter_subsets[i], error_analysis_parameters[i], filepath)
        
    def save_confidence_intervals(self, score_vs_parameters, error_analysis_parameters):
        ''' Saves chi2 as a function of individual fitting parameters '''
        if self.save_data:
            c = 0
            for i in range(len(error_analysis_parameters)):
                for j in range(len(error_analysis_parameters[i])):
                    filepath = self.directory + 'score_vs_parameter_' + str(c+1) + '.dat'
                    save_score_vs_parameter(score_vs_parameters[c], error_analysis_parameters[i][j], filepath)    
                    c += 1      