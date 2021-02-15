'''
Main file of osPDSFit
'''

import argparse
from input.read_config import read_config
from supplement import make_output_directory 


if __name__ == '__main__':
    # Read out the config file 
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help="Path to the configuration file")
    args = parser.parse_args()
    filepath_config = args.filepath
    mode, experiments, spins, simulation_settings, calculation_settings, output_settings = read_config(filepath_config)

    # Make an output directory
    make_output_directory(output_settings, filepath_config)