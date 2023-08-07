import plots.set_matplotlib
import matplotlib.pyplot as plt
from plots.simulation.plot_epr_spectrum import plot_epr_spectrum
from plots.simulation.plot_bandwidths import plot_bandwidths
from plots.simulation.plot_background_time_traces import plot_background_time_traces
from plots.simulation.plot_background_free_time_traces import plot_background_free_time_traces
from plots.simulation.plot_simulated_spectra import plot_simulated_spectra
from plots.simulation.plot_dipolar_angles import plot_dipolar_angles
from plots.simulation.plot_simulated_time_traces import plot_simulated_time_traces
from plots.fitting.plot_score import plot_score, plot_score_multiple_runs
from plots.error_analysis.plot_error_surfaces import plot_error_surfaces
from plots.error_analysis.plot_error_profiles import plot_error_profiles


class Plotter:
    ''' Plots the output data of the program '''
    
    def __init__(self, data_saver=None):
        self.data_saver = data_saver
        self.figure_format = 'png'
        self.dpi = 600   
    
    def plot_simulation_output(self, epr_spectra, bandwidths, simulated_time_traces, background_time_traces, 
                               background_free_time_traces, simulated_spectra, dipolar_angle_distributions, experiments):
        ''' Plots the simulation output '''
        self.plot_bandwidths(bandwidths, experiments, epr_spectra)
        self.plot_simulated_time_traces(simulated_time_traces, [], experiments)
        self.plot_background_time_traces(background_time_traces, [], experiments)
        self.plot_background_free_time_traces(background_free_time_traces, experiments)
        self.plot_simulated_spectra(simulated_spectra, experiments)
        self.plot_dipolar_angle_distributions(dipolar_angle_distributions, experiments)
        
    def plot_fitting_output(self, epr_spectra, bandwidths, idx_best_solution, score_all_runs, simulated_time_traces, background_time_traces, 
                            background_free_time_traces, simulated_spectra, dipolar_angle_distributions, experiments, goodness_of_fit):
        ''' Plots the fitting output '''
        self.plot_bandwidths(bandwidths, experiments, epr_spectra)
        self.plot_score(score_all_runs[idx_best_solution], goodness_of_fit)
        self.plot_score_multiple_runs(score_all_runs, goodness_of_fit, idx_best_solution)
        self.plot_fits(simulated_time_traces, [], experiments)
        self.plot_background_time_traces(background_time_traces, [], experiments)
        self.plot_background_free_time_traces(background_free_time_traces, experiments)
        self.plot_simulated_spectra(simulated_spectra, experiments)
        self.plot_dipolar_angle_distributions(dipolar_angle_distributions, experiments)
    
    def plot_error_analysis_output(self, error_surfaces, error_surfaces_2d, error_profiles, optimized_model_parameters, optimized_model_parameters_all_runs,
                                   error_analysis_parameters, fitting_parameters, model_parameter_uncertainty_interval_bounds, chi2_minimum, chi2_thresholds,
                                   experiments, simulated_time_traces, error_bars_simulated_time_traces, 
                                   background_time_traces, error_bars_background_time_traces,
                                   background_free_time_traces, simulated_spectra, dipolar_angle_distributions):
        ''' Plot the error analysis output '''
        self.plot_error_surfaces(error_surfaces, error_surfaces_2d, optimized_model_parameters, error_analysis_parameters, fitting_parameters, chi2_minimum, chi2_thresholds)
        self.plot_error_surfaces_multiple_runs(error_surfaces, error_surfaces_2d, optimized_model_parameters_all_runs, error_analysis_parameters, fitting_parameters, chi2_minimum, chi2_thresholds)
        self.plot_error_profiles(error_profiles, optimized_model_parameters, error_analysis_parameters, fitting_parameters, 
                                 model_parameter_uncertainty_interval_bounds, chi2_minimum, chi2_thresholds)
        self.plot_fits(simulated_time_traces, error_bars_simulated_time_traces, experiments)
        self.plot_background_time_traces(background_time_traces, error_bars_background_time_traces, experiments)
        self.plot_background_free_time_traces(background_free_time_traces, experiments)
        self.plot_simulated_spectra(simulated_spectra, experiments)
        self.plot_dipolar_angle_distributions(dipolar_angle_distributions, experiments)

    def plot_epr_spectrum(self, spectrum, experiment_name):
        ''' Plots a simulated EPR spectrum '''
        fig = plot_epr_spectrum(spectrum)
        if not (data_saver is None) and self.data_saver.save_figures:
            filepath = self.data_saver.directory + 'epr_spectrum_' + experiment_name + '.' + self.figure_format
            fig.savefig(filepath, format=self.figure_format, dpi=self.dpi)
            plt.close(fig)

    def plot_bandwidths(self, bandwidths, experiments, spectra=[]):
        ''' Plots the bandwidths of detection and pump pulses ''' 
        fig = plot_bandwidths(bandwidths, experiments, spectra)
        if not (self.data_saver is None) and self.data_saver.save_figures:
            filepath = self.data_saver.directory + 'bandwidths' + '.' + self.figure_format
            fig.savefig(filepath, format=self.figure_format, dpi=self.dpi)
            plt.close(fig)
    
    def plot_background_time_traces(self, background_time_traces, error_bars_background_time_traces, experiments):
        ''' Plots the background parts of PDS time traces '''
        fig = plot_background_time_traces(background_time_traces, error_bars_background_time_traces, experiments)
        if not (self.data_saver is None) and self.data_saver.save_figures:
            filepath = self.data_saver.directory + 'backgrounds' + '.' + self.figure_format
            fig.savefig(filepath, format=self.figure_format, dpi=self.dpi)
            plt.close(fig)

    def plot_background_free_time_traces(self, background_free_time_traces, experiments):
        ''' Plots the background-free parts of PDS time traces '''
        fig = plot_background_free_time_traces(background_free_time_traces, experiments)
        if not (self.data_saver is None) and self.data_saver.save_figures:
            filepath = self.data_saver.directory + 'background_free_time_traces' + '.' + self.figure_format
            fig.savefig(filepath, format=self.figure_format, dpi=self.dpi)
            plt.close(fig)
   
    def plot_simulated_spectra(self, simulated_spectra, experiments):
        ''' Plots simulated dipolar spectra '''
        fig = plot_simulated_spectra(simulated_spectra, experiments)
        if not (self.data_saver is None) and self.data_saver.save_figures:
            filepath = self.data_saver.directory + 'dipolar_spectra' + '.' + self.figure_format
            fig.savefig(filepath, format=self.figure_format, dpi=self.dpi)
            plt.close(fig)
        
    def plot_dipolar_angle_distributions(self, dipolar_angle_distributions, experiments):
        ''' Plots simulated distributions of dipolar angles '''
        fig = plot_dipolar_angles(dipolar_angle_distributions, experiments)
        if not (self.data_saver is None) and self.data_saver.save_figures:
            filepath = self.data_saver.directory + 'dipolar_angles' + '.' + self.figure_format
            fig.savefig(filepath, format=self.figure_format, dpi=self.dpi)
            plt.close(fig)

    def plot_simulated_time_traces(self, simulated_time_traces, error_bars_simulated_time_traces, experiments):
        ''' Plots simulated PDS time traces '''
        fig = plot_simulated_time_traces(simulated_time_traces, error_bars_simulated_time_traces, experiments)
        if not (self.data_saver is None) and self.data_saver.save_figures:
            filepath = self.data_saver.directory + 'time_traces' + '.' + self.figure_format
            fig.savefig(filepath, format=self.figure_format, dpi=self.dpi)
            plt.close(fig)

    def plot_score(self, score, goodness_of_fit):
        ''' Plots the score as a function of optimization step '''
        fig = plot_score(score, goodness_of_fit)
        if not (self.data_saver is None) and self.data_saver.save_figures:
            filepath = self.data_saver.directory + 'score' + '.' + self.figure_format
            fig.savefig(filepath, format=self.figure_format, dpi=self.dpi)
            plt.close(fig)
    
    def plot_score_multiple_runs(self, score_all_runs, goodness_of_fit, idx_best_solution):
        ''' Plots the score as a function of optimization step obtained in multiple optimizations '''
        if len(score_all_runs) > 1:
            fig = plot_score_multiple_runs(score_all_runs, goodness_of_fit, idx_best_solution)
            if not (self.data_saver is None) and self.data_saver.save_figures:
                filepath = self.data_saver.directory + 'score_all_runs' + '.' + self.figure_format
                fig.savefig(filepath, format=self.figure_format, dpi=self.dpi)
                plt.close(fig)
    
    def plot_fits(self, simulated_time_traces, error_bars_simulated_time_traces, experiments):
        ''' Plots the fit to PDS time traces '''
        fig = plot_simulated_time_traces(simulated_time_traces, error_bars_simulated_time_traces, experiments)
        if not (self.data_saver is None) and self.data_saver.save_figures:
            filepath = self.data_saver.directory + 'fits' + '.' + self.figure_format
            fig.savefig(filepath, format=self.figure_format, dpi=self.dpi)
            plt.close(fig)

    def plot_error_surfaces(self, error_surfaces, error_surfaces_2d, optimized_model_parameters, error_analysis_parameters, fitting_parameters, chi2_minimum, chi2_thresholds):
        ''' Plots error surfaces '''
        fig = plot_error_surfaces(error_surfaces, error_surfaces_2d, [optimized_model_parameters], error_analysis_parameters, fitting_parameters, chi2_minimum, chi2_thresholds)
        if not (self.data_saver is None) and self.data_saver.save_figures and not (fig is None):
            filepath = self.data_saver.directory + 'error_surfaces' + '.' + self.figure_format
            fig.savefig(filepath, format=self.figure_format, dpi=self.dpi)
            plt.close(fig)
    
    def plot_error_surfaces_multiple_runs(self, error_surfaces, error_surfaces_2d, optimized_model_parameters_all_runs, error_analysis_parameters, fitting_parameters, chi2_minimum, chi2_thresholds):
        ''' Plots error surfaces '''
        fig = plot_error_surfaces(error_surfaces, error_surfaces_2d, optimized_model_parameters_all_runs, error_analysis_parameters, fitting_parameters, chi2_minimum, chi2_thresholds)
        if not (self.data_saver is None) and self.data_saver.save_figures and not (fig is None):
            filepath = self.data_saver.directory + 'error_surfaces_all_runs' + '.' + self.figure_format
            fig.savefig(filepath, format=self.figure_format, dpi=self.dpi)
            plt.close(fig)
    
    def plot_error_profiles(self, error_profiles, optimized_model_parameters, error_analysis_parameters, fitting_parameters, 
                            model_parameter_uncertainty_interval_bounds, chi2_minimum, chi2_thresholds):
        ''' Plots confidence intervals '''
        fig = plot_error_profiles(error_profiles, [optimized_model_parameters], error_analysis_parameters, fitting_parameters, 
                                  model_parameter_uncertainty_interval_bounds, chi2_minimum, chi2_thresholds)
        if not (self.data_saver is None) and self.data_saver.save_figures:
            filepath = self.data_saver.directory + 'error_profiles' + '.' + self.figure_format
            fig.savefig(filepath, format=self.figure_format, dpi=self.dpi)
            plt.close(fig)
    
    def plot_error_profiles_multiple_runs(self, error_profiles, optimized_model_parameters_all_runs, error_analysis_parameters, fitting_parameters, 
                                          model_parameter_uncertainty_interval_bounds, chi2_minimum, chi2_thresholds):
        ''' Plots confidence intervals '''
        fig = plot_error_profiles(error_profiles, optimized_model_parameters_all_runs, error_analysis_parameters, fitting_parameters, 
                                  model_parameter_uncertainty_interval_bounds, chi2_minimum, chi2_thresholds)
        if not (self.data_saver is None) and self.data_saver.save_figures:
            filepath = self.data_saver.directory + 'error_profiles_all_runs' + '.' + self.figure_format
            fig.savefig(filepath, format=self.figure_format, dpi=self.dpi)
            plt.close(fig)
