import sys
import time
import datetime
import argparse
import multiprocessing
try:
    import mpi4py
    from mpi4py import MPI
except:
    pass
from functools import partial
from input.read_config import read_config
from fitting.objective_function import objective_function, fit_function, objective_function_with_background_record
from fitting.check_relative_weights import check_relative_weights
from symmetry.symmetry_related_solutions import compute_symmetry_related_solutions
from output.fitting.print_model_parameters import print_model_parameters
from output.fitting.print_background_parameters import print_background_parameters
from mpi.mpi_status import set_mpi, get_mpi


if __name__ == '__main__':

    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="Path to the configuration file")
    parser.add_argument('--mpisupport', type=int, default=0, help="Activate the MPI support (default: 0)")
    args = parser.parse_args()
    filepath_config = args.config
    mpi_support = args.mpisupport
    set_mpi(mpi_support)
    
    # Init multiprocessing
    if not get_mpi():
        multiprocessing.freeze_support() 
    
    time_start = time.time()
    
    # Read out the config file
    mode, experiments, spins, background, simulator, simulation_parameters, optimizer, fitting_parameters, error_analyzer, error_analysis_parameters, data_saver, plotter = \
        read_config(filepath_config)

    # Set the background model
    simulator.set_background_model(background)

    # Run simulations
    if mode['simulation']:

        # Run precalculations
        simulator.run_precalculations(experiments, spins)
    
        # Simulate the EPR spectrum of the spin system
        epr_spectra = simulator.simulate_epr_spectra(spins, experiments)
        
        # Compute the bandwidths of the detection and pump pulses
        bandwidths = simulator.simulate_bandwidths(experiments)
        
        # Simulate the PDS time traces
        simulated_time_traces, background_parameters, background_time_traces, background_free_time_traces, simulated_spectra, modulation_depths, dipolar_angle_distributions = \
            simulator.simulate_time_traces(experiments, spins, simulation_parameters)
        
        # Save the simulation output
        data_saver.save_simulation_output(epr_spectra, 
                                          bandwidths, 
                                          simulated_time_traces, 
                                          background_time_traces, 
                                          background_free_time_traces, 
                                          simulated_spectra, 
                                          dipolar_angle_distributions,
                                          background_parameters, 
                                          simulator.background, 
                                          experiments)
        
        # Plot the simulation output
        plotter.plot_simulation_output(epr_spectra, 
                                       bandwidths, 
                                       simulated_time_traces, 
                                       background_time_traces, 
                                       background_free_time_traces, 
                                       simulated_spectra, 
                                       dipolar_angle_distributions,
                                       experiments)

    # Run fitting
    if mode['fitting']:
        
        # Run precalculations
        simulator.run_precalculations(experiments, spins)
        
        # Simulate the EPR spectrum of the spin system
        epr_spectra = simulator.simulate_epr_spectra(spins, experiments)
        
        # Compute the bandwidths of the detection and pump pulses
        bandwidths = simulator.simulate_bandwidths(experiments)

        # Set the fit and objective functions
        partial_fit_function = partial(fit_function, 
                                       simulator=simulator, 
                                       experiments=experiments, 
                                       spins=spins,
                                       fitting_parameters=fitting_parameters, 
                                       reset_field_orientations=True,
                                       fixed_variables_included=True, 
                                       more_output=False)
        partial_fit_function_more_output = partial(fit_function, 
                                                   simulator=simulator, 
                                                   experiments=experiments, 
                                                   spins=spins,
                                                   fitting_parameters=fitting_parameters, 
                                                   reset_field_orientations=True,
                                                   fixed_variables_included=True, 
                                                   more_output=True)    
        partial_objective_function = partial(objective_function, 
                                             simulator=simulator, 
                                             experiments=experiments, 
                                             spins=spins,
                                             fitting_parameters=fitting_parameters, 
                                             goodness_of_fit=optimizer.goodness_of_fit,
                                             reset_field_orientations=True,                                             
                                             fixed_variables_included=True)
        optimizer.set_fit_function(partial_fit_function)
        optimizer.set_fit_function_more_output(partial_fit_function_more_output)
        optimizer.set_objective_function(partial_objective_function)
        
        # Optimize the model parameters
        optimized_model_parameters_all_runs, score_all_runs, idx_best_solution = optimizer.optimize(fitting_parameters['ranges'])
        
        # Check / correct the relative weights
        optimized_model_parameters_all_runs = check_relative_weights(optimized_model_parameters_all_runs, fitting_parameters)

        # Compute the fit to the experimental PDS time traces and the optimized values of the background parameters
        simulated_time_traces, optimized_background_parameters, background_time_traces, background_free_time_traces, simulated_spectra, modulation_depths, dipolar_angle_distributions = \
            optimizer.get_fit_more_output()

        # Display the optimized model and background parameters
        print_model_parameters(optimized_model_parameters_all_runs[idx_best_solution], [], fitting_parameters)
        print_background_parameters(optimized_background_parameters, [], experiments, simulator.background)
        
        # Compute symmetry-related sets of fitting parameters
        score_function = partial(objective_function, 
                                 simulator=simulator, 
                                 experiments=experiments, 
                                 spins=spins,
                                 fitting_parameters=fitting_parameters, 
                                 goodness_of_fit=optimizer.goodness_of_fit,
                                 reset_field_orientations=False, 
                                 fixed_variables_included=False)                       
        symmetry_related_solutions = compute_symmetry_related_solutions(optimized_model_parameters_all_runs[idx_best_solution], fitting_parameters, simulator, score_function, spins)

        # Save the fitting output
        data_saver.save_fitting_output(epr_spectra, 
                                       bandwidths, 
                                       idx_best_solution,
                                       score_all_runs, 
                                       optimized_model_parameters_all_runs,
                                       optimized_background_parameters,
                                       symmetry_related_solutions, 
                                       simulated_time_traces, 
                                       background_time_traces, 
                                       background_free_time_traces, 
                                       simulated_spectra,
                                       dipolar_angle_distributions,
                                       fitting_parameters,
                                       experiments,
                                       simulator.background)
        
        # Plot the fitting output
        plotter.plot_fitting_output(epr_spectra, 
                                    bandwidths, 
                                    idx_best_solution,
                                    score_all_runs,
                                    simulated_time_traces, 
                                    background_time_traces, 
                                    background_free_time_traces, 
                                    simulated_spectra,
                                    dipolar_angle_distributions,                                    
                                    experiments, 
                                    optimizer.goodness_of_fit)
        
    # Run error analysis
    if mode['error_analysis'] and error_analysis_parameters != [[]] and error_analysis_parameters != []:
        
        # Load / set the optimized values of the model parameters
        if not mode['fitting']:
            optimized_model_parameters, model_parameter_errors, optimized_model_parameters_all_runs = error_analyzer.load_optimized_model_parameters()
            optimizer.optimized_variables = [optimized_model_parameters]
            idx_best_solution = 0
            optimizer.idx_best_solution = idx_best_solution
        else:
            optimized_model_parameters = optimized_model_parameters_all_runs[idx_best_solution]
         
        # Reset the number of Monte-Carlo samples
        reset_status = simulator.reset_num_samples()
        
        if reset_status or not mode['fitting']:
            # Run precalculations
            simulator.run_precalculations(experiments, spins)
            # Compute / recompute the fit to the experimental PDS time traces and the optimized values of the background parameters
            partial_fit_function_more_output = partial(fit_function, 
                                                       simulator=simulator,
                                                       experiments=experiments, 
                                                       spins=spins,
                                                       fitting_parameters=fitting_parameters,
                                                       reset_field_orientations=False,                                                        
                                                       fixed_variables_included=True, 
                                                       more_output=True)
            optimizer.set_fit_function_more_output(partial_fit_function_more_output)
            simulated_time_traces, optimized_background_parameters, background_time_traces, background_free_time_traces, simulated_spectra, modulation_depths, dipolar_angle_distributions = \
                optimizer.get_fit_more_output()
            # Compute symmetry-related sets of fitting parameters
            score_function = partial(objective_function, 
                                     simulator=simulator, 
                                     experiments=experiments, 
                                     spins=spins,
                                     fitting_parameters=fitting_parameters, 
                                     goodness_of_fit=optimizer.goodness_of_fit,
                                     reset_field_orientations=False, 
                                     fixed_variables_included=False)                       
            symmetry_related_solutions = compute_symmetry_related_solutions(optimized_model_parameters, fitting_parameters, simulator, score_function, spins)
            data_saver.save_symmetry_related_solutions(symmetry_related_solutions, fitting_parameters)
        
        # Set the fit and objective functions
        partial_objective_function = partial(objective_function, 
                                             simulator=simulator, 
                                             experiments=experiments, 
                                             spins=spins, 
                                             fitting_parameters=fitting_parameters, 
                                             goodness_of_fit='chi2',
                                             reset_field_orientations=True, 
                                             fixed_variables_included=True)
        partial_objective_function_with_background_record = partial(objective_function_with_background_record, 
                                                                    simulator=simulator, 
                                                                    experiments=experiments, 
                                                                    spins=spins, 
                                                                    fitting_parameters=fitting_parameters, 
                                                                    goodness_of_fit='chi2', 
                                                                    reset_field_orientations=True, 
                                                                    fixed_variables_included=True)
        error_analyzer.set_objective_function(partial_objective_function)
        error_analyzer.set_objective_function_with_background_record(partial_objective_function_with_background_record)

        # Run the error analysis
        model_parameter_errors, background_parameter_errors, error_surfaces, error_surfaces_2d, error_profiles, \
        model_parameter_uncertainty_interval_bounds, chi2_minimum, chi2_thresholds, error_bars_background_time_traces = \
            error_analyzer.run_error_analysis(error_analysis_parameters, 
                                              optimized_model_parameters,
                                              optimized_background_parameters, 
                                              simulated_time_traces,
                                              background_time_traces, 
                                              simulator.background, 
                                              fitting_parameters, 
                                              modulation_depths)
            
        # Display the optimized and fixed parameters of the geometric and background models
        print_model_parameters(optimized_model_parameters, model_parameter_errors, fitting_parameters)
        print_background_parameters(optimized_background_parameters, background_parameter_errors, experiments, simulator.background)

        # Save the error analysis output             
        data_saver.save_error_analysis_output(optimized_model_parameters, 
                                              model_parameter_errors,                                             
                                              optimized_background_parameters, 
                                              background_parameter_errors,
                                              error_surfaces, 
                                              error_profiles,
                                              simulator.background, 
                                              fitting_parameters, 
                                              error_analysis_parameters, 
                                              experiments,
                                              simulated_time_traces, 
                                              [],
                                              background_time_traces, 
                                              error_bars_background_time_traces,
                                              background_free_time_traces, 
                                              simulated_spectra,
                                              dipolar_angle_distributions)
        
        # Plot the error analysis output 
        plotter.plot_error_analysis_output(error_surfaces,
                                           error_surfaces_2d,
                                           error_profiles, 
                                           optimized_model_parameters,
                                           optimized_model_parameters_all_runs,
                                           error_analysis_parameters, 
                                           fitting_parameters,
                                           model_parameter_uncertainty_interval_bounds,
                                           chi2_minimum,                                       
                                           chi2_thresholds, 
                                           experiments,
                                           simulated_time_traces, 
                                           [], 
                                           background_time_traces, 
                                           error_bars_background_time_traces, 
                                           background_free_time_traces, 
                                           simulated_spectra,
                                           dipolar_angle_distributions)
                                   
    time_finish = time.time()
    time_elapsed = str(datetime.timedelta(seconds = time_finish - time_start))
    sys.stdout.write('\nDONE! Total duration: %s' % (time_elapsed))
    sys.stdout.flush()