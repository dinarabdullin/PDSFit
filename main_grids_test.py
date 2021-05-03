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
    #simulator.precalculations(experiments, spins)
    
    # Run simulations
    if mode['simulation']:
    
        # Simulate the EPR spectrum of the spin system
        #epr_spectra = simulator.epr_spectra(spins, experiments)
        
        #plotter.plot_epr_spectrum(epr_spectra[0], "")

        
        
        # Compute the bandwidths of the detection and pump pulses
        #bandwidths = simulator.bandwidths(experiments)
        
        # Simulate the PDS time traces
        simulated_time_traces, modulation_depth_scale_factors = simulator.compute_time_traces(experiments, spins, simulation_parameters)
        
        
    