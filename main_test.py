''' Main file of PeldorFit: test version'''

import argparse
from datetime import time

from input.read_config import read_config
from simulation.simulator import Simulator
from output.make_output_directory import make_output_directory 
from output.save_epr_spectrum import save_epr_spectrum
from output.save_bandwidths import save_bandwidths
from output.save_time_traces import save_time_traces
from plots.plot_epr_spectrum import plot_epr_spectrum
from plots.plot_bandwidths import plot_bandwidths
from plots.plot_time_traces import plot_time_traces
from plots.keep_figures_visible import keep_figures_visible

if __name__ == '__main__':
    # Read out the config file 
    #parser = argparse.ArgumentParser()
    #parser.add_argument('filepath', help='Path to the configuration file')
    #args = parser.parse_args()
    #filepath_config = args.filepath
    filepath_config = 'examples/example01_nitroxide_biradical_Wband_PELDOR/config_ex01.cfg'
    #filepath_config = 'examples/example01_nitroxide_biradical_Wband_PELDOR/config_ex01_broad_distr.cfg'
    mode, experiments, spins, simulation_settings, calculation_settings, output_settings = read_config(filepath_config)

    # Make an output directory
    make_output_directory(output_settings, filepath_config)

    # Init simulator
    simulator = Simulator(calculation_settings)
    
    # Run calculations
    if mode['simulation']:
    
        # Simulate the EPR spectrum of the spin system
        epr_spectra = simulator.epr_spectra(spins, experiments)
        # Save the spectrum
        save_epr_spectrum(epr_spectra[0], output_settings['directory'], experiments[0].name)
        # Plot the spectrum        
        plot_epr_spectrum(epr_spectra[0], True, output_settings['directory'], experiments[0].name)
        
        # Compute the bandwidths of the detection and pump pulses
        bandwidths = simulator.bandwidths(experiments)
        # Save the bandwidths
        save_bandwidths(bandwidths, experiments, output_settings['directory'])
        # Plot the bandwidths on top of the EPR spectrum of the spin system
        plot_bandwidths(bandwidths, experiments, epr_spectra, True, output_settings['directory'])
        
        # Simulate the PDS time traces
        simulated_time_traces = simulator.compute_time_traces(experiments, spins, simulation_settings['parameters'])
        # Save the time traces
        save_time_traces(simulated_time_traces, experiments, output_settings['directory'])
        # Plot the time traces
        plot_time_traces(simulated_time_traces, experiments, True, output_settings['directory'])
        
    #elif mode['fitting']:
    
    #elif mode['error_analysis']:

    keep_figures_visible()
    print('DONE!')