import argparse
from functools import partial
import multiprocessing
from input.read_config import read_config
from fitting.objective_function import objective_function, fit_function
from output.fitting.print_fitting_parameters import print_fitting_parameters
from output.fitting.print_modulation_depth_scale_factors import print_modulation_depth_scale_factors
from plots.keep_figures_visible import keep_figures_visible


if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Read out the config file 
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help="Path to the configuration file")
    args = parser.parse_args()
    filepath_config = args.filepath
    mode, experiments, spins, simulation_parameters, fitting_parameters, optimizer, \
    error_analysis_parameters, error_analyzer, simulator, data_saver, plotter = read_config(filepath_config)
    
    # Run precalculations
    simulator.precalculations(experiments, spins)
    
    # Run simulations
    if mode['simulation']:
    
        # Simulate the EPR spectrum of the spin system
        epr_spectra = simulator.epr_spectra(spins, experiments)
        
        # Compute the bandwidths of the detection and pump pulses
        bandwidths = simulator.bandwidths(experiments)
        
        # Simulate the PDS time traces
        simulated_time_traces, modulation_depth_scale_factors = simulator.compute_time_traces(experiments, spins, simulation_parameters)
        
        # Save the simulation output
        data_saver.save_simulation_output(epr_spectra, bandwidths, simulated_time_traces, experiments)
        
        # Plot the simulation output
        plotter.plot_simulation_output(epr_spectra, bandwidths, simulated_time_traces, experiments)

    # Run fitting
    elif mode['fitting']:
        
        # Partial functions to calculate the fit and the goodness of fit
        partial_fit_function = partial(fit_function, simulator=simulator, experiments=experiments, spins=spins, fitting_parameters=fitting_parameters)
        partial_objective_function = partial(objective_function, simulator=simulator, experiments=experiments, spins=spins, fitting_parameters=fitting_parameters)
    
        # Optimize the fitting parameters
        optimized_parameters, score = optimizer.optimize(fitting_parameters['ranges'], partial_objective_function)                                                         
        
        # Display the fitted and fixed parameters
        print_fitting_parameters(fitting_parameters['indices'], optimized_parameters, fitting_parameters['values'])
        
        # Compute the fit to the experimental PDS time traces
        simulated_time_traces, modulation_depth_scale_factors = optimizer.get_fit(partial_fit_function)
        
        # Display the scale factors of modulation depths
        if simulator.fit_modulation_depth:
            print_modulation_depth_scale_factors(modulation_depth_scale_factors, experiments)

        # Save the fitting output
        data_saver.save_fitting_output(score, optimized_parameters, [], simulated_time_traces, fitting_parameters, experiments)
        
        # Plot the fitting output
        plotter.plot_fitting_output(score, simulated_time_traces, experiments)
        
        # Prior to the error analysis, check that the calculated chi2 is normalized by the variance of noise.
        # For this, the std of noise must be nonzero for all experiments. 
        # If the std of noise is equal 0 for some (or all) experiments, compute it using the obtained fits as a noise-free signal.
        for i in range(len(experiments)):
            experiments[i].reset_noise_std(simulated_time_traces[i]['s'])
        
        # Re-set the partial function to calculate the fit 
        partial_fit_function = partial(fit_function, simulator=simulator, experiments=experiments, spins=spins, fitting_parameters=fitting_parameters)
        
        # Run the error analysis
        score_vs_parameter_sets, numerical_error, score_threshold, parameters_errors = error_analyzer.run_error_analysis(error_analysis_parameters, 
                                                                                       optimized_parameters, fitting_parameters, partial_objective_function)
        
        # Plot the error analysis output
        plotter.plot_error_analysis_output(error_analysis_parameters, score_vs_parameter_sets, optimized_parameters, fitting_parameters, 
                                           score_threshold, numerical_error)
        
    #elif mode['error_analysis']:
    
    print('\nDONE!')
    keep_figures_visible()