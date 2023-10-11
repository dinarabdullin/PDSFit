import plots.set_matplotlib
import matplotlib.pyplot as plt
from plots.simulation.plot_bandwidths import plot_bandwidths
from plots.simulation.plot_simulated_time_traces import plot_simulated_time_traces
from plots.simulation.plot_backgrounds import plot_backgrounds
from plots.simulation.plot_form_factors import plot_form_factors
from plots.simulation.plot_dipolar_spectra import plot_dipolar_spectra
from plots.simulation.plot_dipolar_angle_distributions import plot_dipolar_angle_distributions
from plots.fitting.plot_score import plot_score, plot_score_all_runs
from plots.error_analysis.plot_error_surfaces import plot_error_surfaces


class Plotter:
    """Plot the output data of PDSFit."""
    
    def __init__(self, data_saver=None):
        self.data_saver = data_saver
        self.figure_format = "png"
        self.dpi = 600   
    
    
    def plot_simulation_output(
            self, epr_spectra, bandwidths, simulated_time_traces, simulated_data, experiments
            ):
        """Plot the simulation output."""
        if not (self.data_saver is None) and self.data_saver.save_figures:
            self.plot_bandwidths(bandwidths, epr_spectra, experiments)
            self.plot_simulated_time_traces(simulated_time_traces, experiments)
            self.plot_backgrounds(simulated_data["background"], experiments)
            self.plot_form_factors(simulated_data["form_factor"], experiments)
            self.plot_dipolar_spectra(simulated_data["dipolar_spectrum"], experiments)
            self.plot_dipolar_angle_distributions(simulated_data["dipolar_angle_distribution"], experiments)
    
    
    def plot_bandwidths(self, bandwidths, epr_spectra, experiments):
        """Plot the bandwidths of detection and pump pulses 
        overlaid with an EPR spectrum of a spin system.""" 
        fig = plot_bandwidths(bandwidths, epr_spectra, experiments)
        filepath = self.data_saver.directory + "bandwidths" + "." + self.figure_format
        fig.savefig(filepath, format = self.figure_format, dpi = self.dpi)
        plt.close(fig)
    
    
    def plot_simulated_time_traces(self, simulated_time_traces, experiments, error_bars = []):
        """Plot experimental and simulated PDS time traces."""
        fig = plot_simulated_time_traces(simulated_time_traces, experiments, error_bars)
        filepath = self.data_saver.directory + "time_traces" + "." + self.figure_format
        fig.savefig(filepath, format = self.figure_format, dpi = self.dpi)
        plt.close(fig)
    
    
    def plot_backgrounds(self, backgrounds, experiments, error_bars = []):
        """Plot experimental PDS time traces and their simulated backgrounds."""
        fig = plot_backgrounds(backgrounds, experiments, error_bars)
        filepath = self.data_saver.directory + "backgrounds" + "." + self.figure_format
        fig.savefig(filepath, format = self.figure_format, dpi = self.dpi)
        plt.close(fig)

    def plot_form_factors(self, form_factors, experiments):
        """Plot form factors for experimental and simulated PDS time traces."""
        fig = plot_form_factors(form_factors, experiments)
        filepath = self.data_saver.directory + "form_factors" + "." + self.figure_format
        fig.savefig(filepath, format = self.figure_format, dpi = self.dpi)
        plt.close(fig)
   
    def plot_dipolar_spectra(self, dipolar_spectra, experiments):
        """Plot dipolar spectra for experimental and simulated PDS time traces."""
        fig = plot_dipolar_spectra(dipolar_spectra, experiments)
        filepath = self.data_saver.directory + "dipolar_spectra" + "." + self.figure_format
        fig.savefig(filepath, format = self.figure_format, dpi = self.dpi)
        plt.close(fig)
        
    def plot_dipolar_angle_distributions(self, dipolar_angle_distributions, experiments):
        """Plot simulated distributions of the dipolar angle."""
        fig = plot_dipolar_angle_distributions(dipolar_angle_distributions, experiments)
        filepath = self.data_saver.directory + "dipolar_angle_distr" + "." + self.figure_format
        fig.savefig(filepath, format = self.figure_format, dpi = self.dpi)
        plt.close(fig)


    def plot_fitting_output(
            self, epr_spectra, bandwidths, score_all_runs, index_best_model, simulated_time_traces, 
            simulated_data, experiments, goodness_of_fit
        ):
        """Plot the fitting output."""
        if not (self.data_saver is None) and self.data_saver.save_figures:
            self.plot_bandwidths(bandwidths, epr_spectra, experiments)
            self.plot_score(score_all_runs, index_best_model, goodness_of_fit)
            self.plot_simulated_time_traces(simulated_time_traces, experiments)
            self.plot_backgrounds(simulated_data["background"], experiments)
            self.plot_form_factors(simulated_data["form_factor"], experiments)
            self.plot_dipolar_spectra(simulated_data["dipolar_spectrum"], experiments)
            self.plot_dipolar_angle_distributions(simulated_data["dipolar_angle_distribution"], experiments)


    def plot_score(self, score_all_runs, index_best_run, goodness_of_fit):
        """Plot goodness-of-fit vs. optimization step."""
        fig = plot_score(score_all_runs[index_best_run], goodness_of_fit)
        filepath = self.data_saver.directory + "score" + "." + self.figure_format
        fig.savefig(filepath, format = self.figure_format, dpi = self.dpi)
        plt.close(fig)
        if len(score_all_runs) > 1:
            fig = plot_score_all_runs(score_all_runs, index_best_run, goodness_of_fit)
            filepath = self.data_saver.directory + "score_all_runs" + "." + self.figure_format
            fig.savefig(filepath, format = self.figure_format, dpi = self.dpi)
            plt.close(fig)


    def plot_error_analysis_output(
        self, best_model, optimized_models, simulated_time_traces, simulated_data, 
        error_analysis_data, experiments, fitting_parameters
        ):
        """Plot the error analysis output."""
        if not (self.data_saver is None) and self.data_saver.save_figures:
            self.plot_simulated_time_traces(simulated_time_traces, experiments)
            self.plot_backgrounds(simulated_data["background"], experiments, error_analysis_data["errors_backgrounds"])
            self.plot_form_factors(simulated_data["form_factor"], experiments)
            self.plot_dipolar_spectra(simulated_data["dipolar_spectrum"], experiments)
            self.plot_dipolar_angle_distributions(simulated_data["dipolar_angle_distribution"], experiments)
            self.plot_error_surfaces(
                error_analysis_data["error_surfaces"] + error_analysis_data["error_surfaces_2d"], 
                error_analysis_data["chi2_minimum"], error_analysis_data["chi2_thresholds"], 
                best_model, optimized_models, fitting_parameters,
                title = "error_surfaces", show_uncertainty_interval = False
                )  
            self.plot_error_surfaces(
                error_analysis_data["error_surfaces_1d"], 
                error_analysis_data["chi2_minimum"], error_analysis_data["chi2_thresholds"],
                best_model, optimized_models, fitting_parameters,
                title = "error_surfaces_1d", show_uncertainty_interval = True
                )

    def plot_error_surfaces(
        self, error_surfaces, chi2_minimum, chi2_thresholds, best_model, optimized_models, fitting_parameters,
        title = "error_surfaces", show_uncertainty_interval = False
        ):
        """Plot error surfaces."""
        fig = plot_error_surfaces(
            error_surfaces, chi2_minimum, chi2_thresholds, [best_model], fitting_parameters, show_uncertainty_interval
            )
        filepath = self.data_saver.directory + title + "." + self.figure_format
        fig.savefig(filepath, format = self.figure_format, dpi = self.dpi)
        plt.close(fig)
        if len(optimized_models) > 1:
            fig = plot_error_surfaces(
                error_surfaces, chi2_minimum, chi2_thresholds, optimized_models, fitting_parameters, show_uncertainty_interval
                )
            filepath = self.data_saver.directory + title + "_all_runs" + "." + self.figure_format
            fig.savefig(filepath, format = self.figure_format, dpi = self.dpi)
            plt.close(fig)