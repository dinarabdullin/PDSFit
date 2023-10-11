import os
import numpy as np
import errno
import datetime
import shutil
from output.simulation.save_epr_spectrum import save_epr_spectrum
from output.simulation.save_bandwidth import save_bandwidth
from output.simulation.save_time_trace import save_time_trace
from output.simulation.save_background import save_background
from output.simulation.save_form_factor import save_form_factor
from output.simulation.save_dipolar_spectrum import save_dipolar_spectrum
from output.simulation.save_dipolar_angle_distribution import save_dipolar_angle_distribution
from output.fitting.save_background_parameters import save_background_parameters
from output.fitting.save_score import save_score, save_score_all_runs 
from output.fitting.save_model_parameters import save_model_parameters, save_model_parameters_all_runs
from output.fitting.save_symmetry_related_models import save_symmetry_related_models
from output.error_analysis.save_error_surface import save_error_surface


class DataSaver:
    """Save the output data of PDSFit."""
    
    def __init__(self, save_data, save_figures):
        self.save_data = save_data
        self.save_figures = save_figures
        self.directory = ""  
    
    
    def create_output_directory(self, parent_directory, filepath_config):
        """Create an output directory.""" 
        if self.save_data or self.save_figures:
            config_directory, config_name = os.path.split(os.path.abspath(filepath_config))
            if parent_directory != "":
                output_directory = parent_directory
            else:
                output_directory = config_directory
                
            now = datetime.datetime.now()
            folder = now.strftime("%Y-%m-%d_%H-%M") + "_" + config_name[:-4]
            output_directory = output_directory + "/" + folder + "/"
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
    
    
    def save_simulation_output(
        self, epr_spectra, bandwidths, simulated_time_traces, simulated_data, experiments, background_model
        ):
        """Save the output of the simulation."""
        if self.save_data:
            self.save_epr_spectra(epr_spectra, experiments)
            self.save_bandwidths(bandwidths, experiments)
            self.save_time_traces(simulated_time_traces, experiments)   
            self.save_backgrounds(simulated_data["background"], experiments)
            self.save_form_factors(simulated_data["form_factor"], experiments)
            self.save_dipolar_spectra(simulated_data["dipolar_spectrum"], experiments)
            self.save_dipolar_angle_distributions(simulated_data["dipolar_angle_distribution"], experiments)
            self.save_background_parameters(simulated_data["background_parameters"], experiments, background_model)
    
    
    def save_epr_spectra(self, epr_spectra, experiments):
        """Save the simulated EPR spectra of a spin system.
        The spectra correspond to magnetic fields, at which PDS time traces were acquired."""
        for i in range(len(experiments)):
            filepath = self.directory + "epr_spectrum_" + experiments[i].name + ".dat"
            save_epr_spectrum(filepath, epr_spectra[i])
    
    
    def save_bandwidths(self, bandwidths, experiments):
        """Save the bandwidths of detection and pump pulses."""
        for i in range(len(experiments)):
            for j in bandwidths[i]:
                filepath = self.directory + j + "_" + experiments[i].name + ".dat"
                save_bandwidth(filepath, bandwidths[i][j])


    def save_time_traces(self, simulated_time_traces, experiments, error_bars = []):
        """Save experimental and simulated PDS time traces."""
        for i in range(len(experiments)):
            filepath = self.directory + "time_trace_" + experiments[i].name + ".dat"
            if len(error_bars) == 0:
                save_time_trace(filepath, simulated_time_traces[i], experiments[i])
            else:
                save_time_trace(filepath, simulated_time_traces[i], experiments[i], error_bars[i])
                    
    
    def save_backgrounds(self, backgrounds, experiments, error_bars = []):
        """Save simulated PDS backgrounds."""
        for i in range(len(experiments)):
            filepath = self.directory + "background_" + experiments[i].name + ".dat"
            if len(error_bars) == 0:
                save_background(filepath, backgrounds[i], experiments[i])
            else:
                save_background(filepath, backgrounds[i], experiments[i], error_bars[i])
                    
      
    def save_form_factors(self, form_factors, experiments):
        """Save simulated PDS form factors."""
        for i in range(len(experiments)):
            filepath = self.directory + "form_factor_" + experiments[i].name + ".dat"
            save_form_factor(filepath, form_factors[i], experiments[i])
    
    
    def save_dipolar_spectra(self, dipolar_spectra, experiments):
        """Save simulated dipolar spectra."""
        for i in range(len(experiments)):
            filepath = self.directory + "dipolar_spectrum_" + experiments[i].name + ".dat"
            save_dipolar_spectrum(filepath, dipolar_spectra[i])
    
    
    def save_dipolar_angle_distributions(self, dipolar_angle_distributions, experiments):
        """Save simulated distributions of the dipolar angle."""
        for i in range(len(experiments)):
            filepath = self.directory + "dipolar_angle_distr_" + experiments[i].name + ".dat"
            save_dipolar_angle_distribution(filepath, dipolar_angle_distributions[i])    
    
    
    def save_background_parameters(self, background_parameters, experiments, background_model, errors = []):
        """Saves the background parameters."""
        filepath = self.directory + "background_parameters.dat"
        if len(errors) != 0:
            save_background_parameters(filepath, background_parameters, experiments, background_model, errors)
        else:
            save_background_parameters(filepath, background_parameters, experiments, background_model)


    def save_fitting_output(
        self, epr_spectra, bandwidths, optimized_models, index_best_model, score_all_runs, simulated_time_traces,
        simulated_data, symmetry_related_models, fitting_parameters, experiments, background_model
        ):
        """Save the output of the fitting procedure."""
        if self.save_data:
            self.save_epr_spectra(epr_spectra, experiments)
            self.save_bandwidths(bandwidths, experiments)
            self.save_score(score_all_runs, index_best_model)
            self.save_model_parameters(optimized_models, index_best_model, fitting_parameters)
            self.save_time_traces(simulated_time_traces, experiments)   
            self.save_backgrounds(simulated_data["background"], experiments)
            self.save_form_factors(simulated_data["form_factor"], experiments)
            self.save_dipolar_spectra(simulated_data["dipolar_spectrum"], experiments)
            self.save_dipolar_angle_distributions(simulated_data["dipolar_angle_distribution"], experiments)
            self.save_background_parameters(simulated_data["background_parameters"], experiments, background_model)
            self.save_symmetry_related_models(symmetry_related_models, fitting_parameters)
    

    def save_score(self, score_all_runs, index_best_trial):
        """Save goodness-of-fit vs. optimization step."""
        filepath = self.directory + "score.dat"
        save_score(filepath, score_all_runs[index_best_trial])
        if len(score_all_runs) > 1:
            for i in range(len(score_all_runs)):
                filepath = self.directory + "score_run" + str(i + 1) + ".dat"
                save_score(filepath, score_all_runs[i])
            filepath = self.directory + "score_all_runs.dat"
            save_score_all_runs(filepath, score_all_runs)


    def save_model_parameters(self, optimized_models, index_best_model, fitting_parameters, errors = []):    
        """Save the parameters of a geometric model.""" 
        filepath = self.directory + "fitting_parameters.dat"
        save_model_parameters(filepath, optimized_models[index_best_model], fitting_parameters, errors)
        if len(optimized_models) > 1:
            for i in range(len(optimized_models)):
                filepath = self.directory + "fitting_parameters_run" + str(i + 1) + ".dat"
                save_model_parameters(filepath, optimized_models[i], fitting_parameters, errors)
            filepath = self.directory + "fitting_parameters_all_runs.dat"
            save_model_parameters_all_runs(filepath, optimized_models, fitting_parameters)


    def save_symmetry_related_models(self, symmetry_related_models, fitting_parameters):
        """Save symmetry-related models."""
        filepath = self.directory + "symmetry_related_models.dat"
        save_symmetry_related_models(filepath, symmetry_related_models, fitting_parameters)


    def save_error_analysis_output(
        self, best_model, simulated_time_traces, simulated_data, symmetry_related_models, error_analysis_data,
        fitting_parameters, experiments, background_model
        ):
        """Save the error analysis output."""
        if self.save_data:
            self.save_model_parameters([best_model], 0, fitting_parameters, error_analysis_data["errors_model_parameters"])
            self.save_time_traces(simulated_time_traces, experiments)
            self.save_backgrounds(
                simulated_data["background"], experiments, error_analysis_data["errors_backgrounds"]
                )
            self.save_form_factors(simulated_data["form_factor"], experiments)
            self.save_dipolar_spectra(simulated_data["dipolar_spectrum"], experiments)
            self.save_dipolar_angle_distributions(simulated_data["dipolar_angle_distribution"], experiments)
            self.save_background_parameters(
                simulated_data["background_parameters"], experiments, background_model, error_analysis_data["errors_background_parameters"]
                )
            self.save_symmetry_related_models(symmetry_related_models, fitting_parameters)
            self.save_error_surfaces(error_analysis_data["error_surfaces"])
            self.save_error_surfaces(error_analysis_data["error_surfaces_2d"], title = "error_surface_2d")
            self.save_error_surfaces(error_analysis_data["error_surfaces_1d"], title = "error_surface_1d")


    def save_error_surfaces(self, error_surfaces, title = None):
        """Save error surfaces."""
        for i in range(len(error_surfaces)):
            if title is None:
                filepath = self.directory + "error_surface" + "_" + str(i + 1) + ".dat"
            else:
                filepath = self.directory + title + "_" + str(i + 1) + ".dat"
            save_error_surface(filepath, error_surfaces[i])