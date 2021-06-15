import plots.set_matplotlib
import matplotlib.pyplot as plt
from plots.simulation.plot_epr_spectrum import plot_epr_spectrum
from plots.simulation.plot_bandwidths import plot_bandwidths
from plots.simulation.plot_simulated_time_traces import plot_simulated_time_traces
from plots.fitting.plot_score import plot_score
from plots.error_analysis.plot_error_surfaces import plot_error_surfaces
from plots.error_analysis.plot_confidence_intervals import plot_confidence_intervals

class Plotter:
    
    def __init__(self, data_saver=None):
        self.data_saver = data_saver
        self.figure_format = 'png'
        self.dpi = 600   

    def plot_epr_spectrum(self, spectrum, experiment_name):
        ''' Plots a simulated EPR spectrum '''
        fig = plot_epr_spectrum(spectrum)
        if not (data_saver is None) and self.data_saver.save_figures:
            filepath = self.data_saver.directory + 'epr_spectrum_' + experiment_name + '.' + self.figure_format
            fig.savefig(filepath, format=self.figure_format, dpi=self.dpi)

    def plot_bandwidths(self, bandwidths, experiments, spectra=[]):
        ''' 
        Plots the bandwidths of detection and pump pulses for multiple experiments.
        If the EPR spectrum of the spin system is provided, the bandwidths are overlayed with the EPR spectrum. 
        ''' 
        fig = plot_bandwidths(bandwidths, experiments, spectra)
        if not (self.data_saver is None) and self.data_saver.save_figures:
            filepath = self.data_saver.directory + 'bandwidths' + '.' + self.figure_format
            fig.savefig(filepath, format=self.figure_format, dpi=self.dpi)

    def plot_simulated_time_traces(self, simulated_time_traces, experiments):
        ''' Plots simulated PDS time traces '''
        fig = plot_simulated_time_traces(simulated_time_traces, experiments)
        if not (self.data_saver is None) and self.data_saver.save_figures:
            filepath = self.data_saver.directory + 'time_traces' + '.' + self.figure_format
            fig.savefig(filepath, format=self.figure_format, dpi=self.dpi)

    def plot_simulation_output(self, epr_spectra, bandwidths, simulated_time_traces, experiments):
        ''' Plots the simulation output '''       
        # self.plot_epr_spectrum(epr_spectra[0], experiments[0].name)
        self.plot_bandwidths(bandwidths, experiments, epr_spectra)
        self.plot_simulated_time_traces(simulated_time_traces, experiments)
    
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

    def plot_fitting_output(self, score, goodness_of_fit, simulated_time_traces, experiments):
        ''' Plots the fitting output '''
        self.plot_score(score, goodness_of_fit)
        self.plot_fits(simulated_time_traces, experiments)
    
    def plot_error_surfaces(self, score_vs_parameter_subsets, error_analysis_parameters, 
                            fitting_parameters, optimized_parameters, score_threshold):
        ''' Plots the score vs the subsets of fitting parameters '''
        fig = plot_error_surfaces(score_vs_parameter_subsets, error_analysis_parameters, 
                                  fitting_parameters, optimized_parameters, score_threshold)
        if not (self.data_saver is None) and self.data_saver.save_figures:
            filepath = self.data_saver.directory + 'score_vs_parameters' + '.' + self.figure_format
            fig.savefig(filepath, format=self.figure_format, dpi=self.dpi)
    
    def plot_confidence_intervals(self, score_vs_parameters, error_analysis_parameters, fitting_parameters, 
                                  optimized_parameters, score_threshold, numerical_error):
        ''' Plots the confidence intervals of individual fitting parameters '''
        fig = plot_confidence_intervals(score_vs_parameters, error_analysis_parameters, fitting_parameters, 
                                       optimized_parameters, score_threshold, numerical_error)
        if not (self.data_saver is None) and self.data_saver.save_figures:
            filepath = self.data_saver.directory + 'confidence_intervals' + '.' + self.figure_format
            fig.savefig(filepath, format=self.figure_format, dpi=self.dpi)
    
    def plot_error_analysis_output(self, score_vs_parameter_subsets, score_vs_parameters, error_analysis_parameters, 
                                   fitting_parameters, optimized_parameters, score_threshold, numerical_error):
        ''' Plot the error analysis output '''
        self.plot_error_surfaces(score_vs_parameter_subsets, error_analysis_parameters, fitting_parameters, optimized_parameters, score_threshold)
        self.plot_confidence_intervals(score_vs_parameters, error_analysis_parameters, fitting_parameters, optimized_parameters, score_threshold, numerical_error)