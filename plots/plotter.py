import plots.set_matplotlib
import matplotlib.pyplot as plt
from plots.simulation.plot_epr_spectrum import plot_epr_spectrum
from plots.simulation.plot_bandwidths import plot_bandwidths
from plots.simulation.plot_background_time_traces import plot_background_time_traces
from plots.simulation.plot_background_free_time_traces import plot_background_free_time_traces
from plots.simulation.plot_simulated_spectra import plot_simulated_spectra
from plots.simulation.plot_simulated_time_traces import plot_simulated_time_traces
from plots.fitting.plot_score import plot_score
from plots.error_analysis.plot_error_surfaces import plot_error_surfaces
from plots.error_analysis.plot_confidence_intervals import plot_confidence_intervals

class Plotter:
    
    def __init__(self, data_saver=None):
        self.data_saver = data_saver
        self.figure_format = 'png'
        self.dpi = 600   
    
    def plot_simulation_output(self, epr_spectra, bandwidths, background_time_traces, background_free_time_traces, 
                               simulated_spectra, simulated_time_traces, experiments):
        ''' Plots the simulation output '''
        self.plot_bandwidths(bandwidths, experiments, epr_spectra)
        self.plot_background_time_traces(background_time_traces, experiments)
        self.plot_background_free_time_traces(background_free_time_traces, experiments)
        self.plot_simulated_spectra(simulated_spectra, experiments)
        self.plot_simulated_time_traces(simulated_time_traces, experiments)
    
    def plot_fitting_output(self, epr_spectra, bandwidths, score, goodness_of_fit, background_time_traces, 
                            background_free_time_traces, simulated_spectra, simulated_time_traces, experiments):
        ''' Plots the fitting output '''
        self.plot_bandwidths(bandwidths, experiments, epr_spectra)
        self.plot_score(score, goodness_of_fit)
        self.plot_background_time_traces(background_time_traces, experiments)
        self.plot_background_free_time_traces(background_free_time_traces, experiments)
        self.plot_simulated_spectra(simulated_spectra, experiments)
        self.plot_fits(simulated_time_traces, experiments)
    
    def plot_error_analysis_output(self, score_vs_parameter_subsets, score_vs_parameters, error_analysis_parameters, 
                                   fitting_parameters, optimized_parameters, score_threshold, numerical_error):
        ''' Plot the error analysis output '''
        self.plot_error_surfaces(score_vs_parameter_subsets, error_analysis_parameters, fitting_parameters, optimized_parameters, score_threshold)
        self.plot_confidence_intervals(score_vs_parameters, error_analysis_parameters, fitting_parameters, optimized_parameters, score_threshold, numerical_error)

    def plot_epr_spectrum(self, spectrum, experiment_name):
        ''' Plots a simulated EPR spectrum '''
        fig = plot_epr_spectrum(spectrum)
        if not (data_saver is None) and self.data_saver.save_figures:
            filepath = self.data_saver.directory + 'epr_spectrum_' + experiment_name + '.' + self.figure_format
            fig.savefig(filepath, format=self.figure_format, dpi=self.dpi)

    def plot_bandwidths(self, bandwidths, experiments, spectra=[]):
        ''' Plots the bandwidths of detection and pump pulses ''' 
        fig = plot_bandwidths(bandwidths, experiments, spectra)
        if not (self.data_saver is None) and self.data_saver.save_figures:
            filepath = self.data_saver.directory + 'bandwidths' + '.' + self.figure_format
            fig.savefig(filepath, format=self.figure_format, dpi=self.dpi)
    
    def plot_background_time_traces(self, background_time_traces, experiments):
        ''' Plots the background parts of PDS time traces '''
        fig = plot_background_time_traces(background_time_traces, experiments)
        if not (self.data_saver is None) and self.data_saver.save_figures:
            filepath = self.data_saver.directory + 'backgrounds' + '.' + self.figure_format
            fig.savefig(filepath, format=self.figure_format, dpi=self.dpi)

    def plot_background_free_time_traces(self, background_free_time_traces, experiments):
        ''' Plots the background-free parts of PDS time traces '''
        fig = plot_background_free_time_traces(background_free_time_traces, experiments)
        if not (self.data_saver is None) and self.data_saver.save_figures:
            filepath = self.data_saver.directory + 'background_free_time_traces' + '.' + self.figure_format
            fig.savefig(filepath, format=self.figure_format, dpi=self.dpi)
   
    def plot_simulated_spectra(self, simulated_spectra, experiments):
        ''' Plots simulated PDS spectra '''
        fig = plot_simulated_spectra(simulated_spectra, experiments)
        if not (self.data_saver is None) and self.data_saver.save_figures:
            filepath = self.data_saver.directory + 'spectra' + '.' + self.figure_format
            fig.savefig(filepath, format=self.figure_format, dpi=self.dpi)

    def plot_simulated_time_traces(self, simulated_time_traces, experiments):
        ''' Plots simulated PDS time traces '''
        fig = plot_simulated_time_traces(simulated_time_traces, experiments)
        if not (self.data_saver is None) and self.data_saver.save_figures:
            filepath = self.data_saver.directory + 'time_traces' + '.' + self.figure_format
            fig.savefig(filepath, format=self.figure_format, dpi=self.dpi)

    def plot_score(self, score, goodness_of_fit):
        ''' Plots the score as a function of optimization step '''
        fig = plot_score(score, goodness_of_fit)
        if not (self.data_saver is None) and self.data_saver.save_figures:
            filepath = self.data_saver.directory + 'score' + '.' + self.figure_format
            fig.savefig(filepath, format=self.figure_format, dpi=self.dpi)

    def plot_fits(self, simulated_time_traces, experiments):
        ''' Plots the fits to experimental PDS time traces '''
        fig = plot_simulated_time_traces(simulated_time_traces, experiments)
        if not (self.data_saver is None) and self.data_saver.save_figures:
            filepath = self.data_saver.directory + 'fits' + '.' + self.figure_format
            fig.savefig(filepath, format=self.figure_format, dpi=self.dpi)

    def plot_error_surfaces(self, score_vs_parameter_subsets, error_analysis_parameters, 
                            fitting_parameters, optimized_parameters, score_threshold):
        ''' Plots chi2 as a function of fitting parameters' subsets '''
        fig = plot_error_surfaces(score_vs_parameter_subsets, error_analysis_parameters, 
                                  fitting_parameters, optimized_parameters, score_threshold)
        if not (self.data_saver is None) and self.data_saver.save_figures:
            filepath = self.data_saver.directory + 'score_vs_parameters' + '.' + self.figure_format
            fig.savefig(filepath, format=self.figure_format, dpi=self.dpi)
    
    def plot_confidence_intervals(self, score_vs_parameters, error_analysis_parameters, fitting_parameters, 
                                  optimized_parameters, score_threshold, numerical_error):
        ''' Plots chi2 as a function of individual fitting parameters '''
        fig = plot_confidence_intervals(score_vs_parameters, error_analysis_parameters, fitting_parameters, 
                                       optimized_parameters, score_threshold, numerical_error)
        if not (self.data_saver is None) and self.data_saver.save_figures:
            filepath = self.data_saver.directory + 'confidence_intervals' + '.' + self.figure_format
            fig.savefig(filepath, format=self.figure_format, dpi=self.dpi)
    
