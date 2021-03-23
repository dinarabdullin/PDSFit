import os
import errno
import datetime
import shutil
from output.simulation.save_epr_spectrum import save_epr_spectrum
from output.simulation.save_bandwidth import save_bandwidth
from output.simulation.save_simulated_time_trace import save_simulated_time_trace
from output.fitting.save_score import save_score
from output.fitting.save_fitting_parameters import save_fitting_parameters


class DataSaver:
    ''' The class to save the program's output data '''
    
    def __init__(self, save_data, save_figures):
        self.save_data = save_data
        self.save_figures = save_figures
        self.directory = ''
    
    def create_output_directory(self, parent_directory, filepath_config):
        ''' Create an output directory ''' 
        if self.save_data or self.save_figures:
            config_directory, config_name = os.path.split(os.path.abspath(filepath_config))
            if parent_directory != '':
                output_directory = parent_directory
            else:
                output_directory = config_directory
                
            now = datetime.datetime.now()
            folder = now.strftime("%Y-%m-%d_%H-%M")
            output_directory = output_directory + "/" + folder + "/"
            try:
                os.makedirs(output_directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise      
            shutil.copy2(filepath_config, output_directory + config_name)
            self.directory = output_directory
    
    def save_epr_spectrum(self, spectrum, experiment_name):
        ''' Saves a simulated EPR spectrum '''
        if self.save_data:
            filepath = self.directory + 'epr_spectrum_' + experiment_name + '.dat'
            save_epr_spectrum(spectrum, filepath)

    def save_bandwidths(self, bandwidths, experiments):
        ''' Saves the bandwidths of detection and pump pulses'''
        if self.save_data:
            for i in range(len(experiments)):
                for key in bandwidths[i]:
                    filepath = self.directory + key + '_' + experiments[i].name + ".dat"
                    save_bandwidth(bandwidths[i][key], filepath)

    def save_simulated_time_traces(self, simulated_time_traces, experiments):
        ''' Saves simulated PDS time traces '''
        if self.save_data:
            for i in range(len(experiments)):
                filepath = self.directory + 'time_trace_' + experiments[i].name + ".dat"
                save_simulated_time_trace(simulated_time_traces[i], experiments[i].s, filepath)

    def save_simulation_output(self, epr_spectra, bandwidths, simulated_time_traces, experiments):
        ''' Saves the simulation output '''
        # self.save_epr_spectrum(epr_spectra[0], experiments[0].name)
        self.save_bandwidths(bandwidths, experiments)
        self.save_simulated_time_traces(simulated_time_traces, experiments)
    
    def save_score(self, score):
        ''' Saves the score as a function of optimization step '''
        if self.save_data:
            filepath = self.directory + 'score.dat'
            save_score(score, filepath)

    def save_fitting_parameters(self, parameters_indices, optimized_parameters, fixed_parameters_values, parameters_errors):    
        ''' Saves optimized and fixed fitting parameters ''' 
        if self.save_data:
            filepath = self.directory + 'fitting_parameters.dat'
            save_fitting_parameters(parameters_indices, optimized_parameters, fixed_parameters_values, parameters_errors, filepath)

    def save_fits(self, simulated_time_traces, experiments):
        ''' Saves fits to experimental PDS time traces '''
        if self.save_data:
            for i in range(len(experiments)):
                filepath = self.directory + 'fit_' + experiments[i].name + ".dat"
                save_simulated_time_trace(simulated_time_traces[i], experiments[i].s, filepath)
    
    def save_fitting_output(self, score, optimized_parameters, parameters_errors, simulated_time_traces, fitting_parameters, experiments):
        ''' Saves the fitting output '''
        self.save_score(score)
        self.save_fitting_parameters(fitting_parameters['indices'], optimized_parameters, fitting_parameters['values'], parameters_errors)
        self.save_fits(simulated_time_traces, experiments)