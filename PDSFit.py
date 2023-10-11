import sys
import time
import datetime
import argparse
from functools import partial
from mpi.mpi_status import set_mpi
from input.read_config import read_config
from fitting.set_bounds import set_bounds
from fitting.scoring_function import scoring_function, fit_function, normalize_weights
from output.fitting.save_model_parameters import print_model_parameters
from output.fitting.save_background_parameters import print_background_parameters
from symmetry.compute_symmetry_related_models import compute_symmetry_related_models


if __name__ == "__main__":
    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help = "Path to the configuration file")
    parser.add_argument("--mpisupport", type = int, default = 0, help = "Activate the MPI support (0 = no, 1 = yes, default: 0)")
    args = parser.parse_args()
    filepath_config = args.config
    mpi_support = args.mpisupport
    set_mpi(mpi_support)
    
    time_start = time.time()
    # Read out the config file
    input_data = read_config(filepath_config)
    mode = input_data["mode"]
    experiments = input_data["experiments"]
    spins = input_data["spins"]
    simulator = input_data["simulator"]
    simulation_parameters = input_data["simulation_parameters"]
    optimizer = input_data["optimizer"]
    fitting_parameters = input_data["fitting_parameters"]
    error_analyzer = input_data["error_analyzer"]
    error_analysis_parameters = input_data["error_analysis_parameters"]
    data_saver = input_data["data_saver"]
    plotter = input_data["plotter"]
    
    # Run simulations
    if mode["simulation"]:
        # Run pre-calculations
        simulator.run_precalculations(experiments, spins)
        # Simulate the EPR spectrum of the spin system
        epr_spectra = simulator.simulate_epr_spectra(spins, experiments)
        # Compute the pulse bandwidths
        bandwidths = simulator.simulate_bandwidths(experiments)
        # Simulate the PDS time traces
        data_labels = ["background_parameters", "background", "form_factor", "dipolar_spectrum", "dipolar_angle_distribution"]
        simulated_time_traces, simulated_data = simulator.simulate_time_traces(
            simulation_parameters, experiments, spins, more_output = data_labels
            )
        # Save the simulation output
        data_saver.save_simulation_output(
            epr_spectra, bandwidths, simulated_time_traces, simulated_data, experiments, simulator.background_model
            )
        # Plot the simulation output
        plotter.plot_simulation_output(
            epr_spectra, bandwidths, simulated_time_traces, simulated_data, experiments
            )

    # Run fitting
    if mode["fitting"]:
        # Run pre-calculations
        simulator.run_precalculations(experiments, spins)
        # Simulate the EPR spectrum of the spin system
        epr_spectra = simulator.simulate_epr_spectra(spins, experiments)
        # Compute the pulse bandwidths
        bandwidths = simulator.simulate_bandwidths(experiments)
        # Set the scoring function  
        scoring_func = partial(
            scoring_function,
            goodness_of_fit = optimizer.goodness_of_fit,
            fitting_parameters = fitting_parameters,            
            simulator = simulator, 
            experiments = experiments, 
            spins = spins,
            more_output = [],
            reset_field_orientations = True,
            fixed_params_included = True
            )
        optimizer.set_scoring_function(scoring_func)
        # Set bounds for fitting parameters
        bounds = set_bounds(fitting_parameters)
        # Optimize fitting parameters
        optimized_models, index_best_model, score_all_runs = optimizer.optimize(bounds)
        optimized_models = normalize_weights(optimized_models, fitting_parameters)
        best_model = optimized_models[index_best_model]
        # Simulate PDS time traces for the best optimized model
        data_labels = ["background_parameters", "background", "form_factor", "dipolar_spectrum", "dipolar_angle_distribution"]
        simulated_time_traces, simulated_data = fit_function(
            best_model, fitting_parameters, simulator, experiments, spins, 
            more_output = data_labels, reset_field_orientations = False, fixed_params_included = True 
            )
        # Display the optimized parameters of a geometric model and a background model
        print_model_parameters(best_model, fitting_parameters)
        print_background_parameters(simulated_data["background_parameters"], experiments, simulator.background_model) 
        # Compute symmetry-related models
        scoring_func_symmetry = partial(
            scoring_function,
            goodness_of_fit = optimizer.goodness_of_fit,
            fitting_parameters = fitting_parameters,            
            simulator = simulator, 
            experiments = experiments, 
            spins = spins,
            more_output = [],
            reset_field_orientations = False,
            fixed_params_included = False
            )
        symmetry_related_models = compute_symmetry_related_models(
            best_model, fitting_parameters, simulator, spins, scoring_func_symmetry
            )
        # Save the fitting output
        data_saver.save_fitting_output(
            epr_spectra, bandwidths, optimized_models, index_best_model, score_all_runs, simulated_time_traces, 
            simulated_data, symmetry_related_models, fitting_parameters, experiments, simulator.background_model
            )
        # Plot the fitting output
        plotter.plot_fitting_output(
            epr_spectra, bandwidths, score_all_runs, index_best_model, simulated_time_traces, 
            simulated_data, experiments, optimizer.goodness_of_fit
            )
        
    # Run error analysis
    if mode["error_analysis"] and len(error_analysis_parameters) != 0:
        if not mode["fitting"]:
            # Load the optimized model(s)
            best_model, errors, optimized_models = error_analyzer.load_optimized_models()
            # Run precalculations
            simulator.run_precalculations(experiments, spins)
            # Simulate PDS time traces for the best optimized model
            data_labels = ["background_parameters", "background", "form_factor", "dipolar_spectrum", "dipolar_angle_distribution"]
            simulated_time_traces, simulated_data = fit_function(
                best_model, fitting_parameters, simulator, experiments, spins, 
                more_output = data_labels, reset_field_orientations = False, fixed_params_included = True
                )
            # Display the optimized parameters of a geometric model and a background model
            print_model_parameters(best_model, fitting_parameters)
            print_background_parameters(simulated_data["background_parameters"], experiments, simulator.background_model) 
            # Compute symmetry-related models
            scoring_func_symmetry = partial(
                scoring_function,
                goodness_of_fit = "chi2",
                fitting_parameters = fitting_parameters,            
                simulator = simulator, 
                experiments = experiments, 
                spins = spins,
                more_output = [],
                reset_field_orientations = False,
                fixed_params_included = False
                )
            symmetry_related_models = compute_symmetry_related_models(
                best_model, fitting_parameters, simulator, spins, scoring_func_symmetry
            )
        # Set the scoring functions
        scoring_func_error_analysis = partial(
            scoring_function,
            goodness_of_fit = "chi2",
            fitting_parameters = fitting_parameters,            
            simulator = simulator, 
            experiments = experiments, 
            spins = spins,
            more_output = ["background_parameters", "modulation_depth"], 
            reset_field_orientations = True,
            fixed_params_included = True
            )
        error_analyzer.set_scoring_function(scoring_func_error_analysis)
        # Perform the error analysis
        error_analysis_data = error_analyzer.run_error_analysis(
            error_analysis_parameters, best_model, simulated_data, fitting_parameters, simulator.background_model, experiments
            )
        # Display the optimized parameters of a geometric model and a background model
        print_model_parameters(best_model, fitting_parameters, error_analysis_data["errors_model_parameters"])
        print_background_parameters(
            simulated_data["background_parameters"], experiments, simulator.background_model, error_analysis_data["errors_background_parameters"]
            )
        # Save the error analysis output             
        data_saver.save_error_analysis_output(
            best_model, simulated_time_traces, simulated_data, symmetry_related_models, error_analysis_data,
            fitting_parameters, experiments, simulator.background_model
            )
        # Plot the error analysis output
        plotter.plot_error_analysis_output(
            best_model, optimized_models, simulated_time_traces, simulated_data, error_analysis_data, experiments, fitting_parameters
            )
                                   
    time_finish = time.time()
    time_elapsed = str(datetime.timedelta(seconds = time_finish - time_start))
    sys.stdout.write("\nDONE! Total duration: %s" % (time_elapsed))
    sys.stdout.flush()