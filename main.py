''' Main file of PeldorFit '''

import argparse
import sys
import multiprocessing
from input.read_config import read_config
from simulation.simulator import Simulator
from fitting.optimizer import Optimizer
from fitting.scoring_function import scoring_function, fit_function
from output.make_output_directory import make_output_directory
from output.logger import Logger, ContextManager
from output.save_simulation_output import save_simulation_output
from output.print_fitting_output import print_fitting_parameters, print_modulation_depth_scale_factors
from output.save_fitting_output import save_fitting_output
from plots.plot_simulation_output import plot_simulation_output
from plots.plot_fitting_output import plot_fitting_output
from plots.keep_figures_visible import keep_figures_visible

if __name__ == '__main__':
    # Add support for when a program which uses multiprocessing has been frozen to produce a Windows executable
    # Source: https://docs.python.org/3/library/multiprocessing.html
    multiprocessing.freeze_support()

    # Read out the config file 
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help="Path to the configuration file")
    args = parser.parse_args()
    filepath_config = args.filepath
    mode, experiments, spins, simulation_parameters, fitting_parameters, \
    optimizer, error_analysis_settings, calculation_settings, output_settings = read_config(filepath_config) 
    
    # Make an output directory
    make_output_directory(output_settings, filepath_config)
    
    # Make a log file
    sys.stdout = ContextManager(output_settings['directory']+'logfile.log')

    # Init simulator
    simulator = Simulator(calculation_settings)
    
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
        if output_settings['save_data']:
            save_simulation_output(epr_spectra, bandwidths, simulated_time_traces, experiments, output_settings['directory'])
        
        # Plot the simulation output
        plot_simulation_output(epr_spectra, bandwidths, simulated_time_traces, experiments, output_settings['save_figures'], output_settings['directory'])

    # Run fitting
    elif mode['fitting']:

        # Optimize the fitting parameters
        optimized_parameters, goodness_of_fit = optimizer.optimize(scoring_function, fitting_parameters['ranges'], simulator=simulator, 
                                                                   experiments=experiments, spins=spins, fitting_parameters=fitting_parameters)
        
        # Display the fitted and fixed parameters
        print_fitting_parameters(fitting_parameters['indices'], optimized_parameters, fitting_parameters['values'])
        
        # Compute the fit to the experimental PDS time traces
        simulated_time_traces, modulation_depth_scale_factors = optimizer.get_fit(fit_function, simulator=simulator, experiments=experiments, 
                                                                                  spins=spins, fitting_parameters=fitting_parameters)
        
        # Display the scale factors of modulation depths
        if simulator.fit_modulation_depth:
            print_modulation_depth_scale_factors(modulation_depth_scale_factors, experiments)

        # Save the fitting output
        if output_settings['save_data']:
            save_fitting_output(goodness_of_fit, optimized_parameters, [], fitting_parameters, simulated_time_traces, experiments, output_settings['directory'])
        
        # Plot the fitting output
        plot_fitting_output(goodness_of_fit, simulated_time_traces, experiments, output_settings['save_figures'], output_settings['directory'])
        
    #elif mode['error_analysis']:
    
    print('\nDONE!')
    keep_figures_visible()