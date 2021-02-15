'''
Read input data from a configuration file
'''

import io
import sys
import libconf
import numpy as np

sys.path.append('..')
from input.read_array import read_array
from input.read_list import read_list
from experiments.experiment_types import experiment_types
from spin_physics.spin import Spin
from supplement.definitions import const

def read_calculation_mode(config):
    mode = {}
    switch = int(config.mode)
    if (switch == 0):
        mode['simulation'] = 1
        mode['fitting'] = 0
        mode['error_analysis'] = 0
    elif (switch == 1):
        mode['simulation'] = 0
        mode['fitting'] = 1
        mode['error_analysis'] = 0
    elif (switch == 2):
        mode['simulation'] = 0
        mode['fitting'] = 0
        mode['error_analysis'] = 1
    else:
        raise ValueError('Invalid mode!')
        sys.exit(1)
    return mode

def read_experimental_parameters(config):
    experiments = []
    for instance in config.experiments:
        name = instance.name
        technique = instance.technique
        detection_frequency = float(instance.detection_frequency)
        detection_pulse_lengths = []
        for pulse_length in instance.detection_pulse_lengths:
            detection_pulse_lengths.append(float(pulse_length)) 
        pump_frequency = float(instance.pump_frequency)
        pump_pulse_lengths = []
        for pulse_length in instance.pump_pulse_lengths:
            pump_pulse_lengths.append(float(pulse_length))
        magnetic_field = float(instance.magnetic_field)
        temperature = float(instance.temperature)
        if technique in experiment_types:
            exp = experiment_types[technique](name, technique, detection_frequency, detection_pulse_lengths, pump_frequency, pump_pulse_lengths, magnetic_field, temperature)
            exp.signal_from_file(instance.filename, 1)
            experiments.append(exp)
        else:
            raise ValueError('Invalid name of experiment!')
            sys.exit(1)   
    return experiments

def read_array(array_obj, data_type, scale=1.0):
    array = []
    data_types = {"float": float, "int": int, "str": str, "list": list}
    if (array_obj != []):
        for c in array_obj:
            if (data_type == "float") or (data_type == "int"):
                array.append(data_types[data_type](c * scale))
            else:
                array.append(data_types[data_type](c))
    return array
    
def read_spin_parameters(config):
    spins = []
    for instance in config.spins:
        g = np.array(read_array(instance.g, "float"))
        if g.size != 3:
            raise ValueError('Invalid number of elements in g!')
            sys.exit(1)
        gStrain = np.array(read_array(instance.gStrain, "float"))
        if gStrain.size != 0 and gStrain.size != 3:
            raise ValueError('Invalid number of elements in gStrain!')
            sys.exit(1)
        n = np.array(read_list(instance.n, "int"))
        I = np.array(read_list(instance.I, "float"))
        if I.size != n.size:
            raise ValueError('Number of elements in n and I must be equal!')
            sys.exit(1)
        if n.size != 0:
            A = np.array(read_list(instance.A, ("array","float")))
            if A.size != 3 * n.size:
                raise ValueError('Invalid number of elements in A!')
                sys.exit(1)
        else:
            A = np.array([])      
        if A.size != 0:
            AStrain = np.array(read_list(instance.AStrain, ("array","float")))
            if AStrain.size != 0 and AStrain.size != A.size: 
                raise ValueError('Invalid number of elements in AStrain!')
                sys.exit(1)
        else:
            AStrain = np.array([]) 
        lwpp = float(instance.lwpp)  
        gAnisotropy = bool(instance.gAnisotropy)
        spin = Spin(g, n, I, A, gStrain, AStrain, lwpp, gAnisotropy)
        spins.append(spin)
    return spins

def read_simulation_settings(config):
    simulation_settings = {}
    simulation_settings['parameters'] = {}
    simulation_settings['parameters']['r_mean'] = read_list(config.simulation_parameters.r_mean, ("array", "float"))
    simulation_settings['parameters']['r_width'] = read_list(config.simulation_parameters.r_width, ("array", "float"))
    simulation_settings['parameters']['xi_mean'] = read_list(config.simulation_parameters.xi_mean, ("array", "float"), const['deg2rad'])
    simulation_settings['parameters']['xi_width'] = read_list(config.simulation_parameters.xi_width, ("array", "float"), const['deg2rad'])
    simulation_settings['parameters']['phi_mean'] = read_list(config.simulation_parameters.phi_mean, ("array", "float"), const['deg2rad'])
    simulation_settings['parameters']['phi_width'] = read_list(config.simulation_parameters.phi_width, ("array", "float"), const['deg2rad'])
    simulation_settings['parameters']['alpha_mean'] = read_list(config.simulation_parameters.alpha_mean, ("array", "float"), const['deg2rad'])
    simulation_settings['parameters']['alpha_width'] = read_list(config.simulation_parameters.alpha_width, ("array", "float"), const['deg2rad'])
    simulation_settings['parameters']['beta_mean'] = read_list(config.simulation_parameters.beta_mean, ("array", "float"), const['deg2rad'])
    simulation_settings['parameters']['beta_width'] = read_list(config.simulation_parameters.beta_width, ("array", "float"), const['deg2rad'])
    simulation_settings['parameters']['gamma_mean'] = read_list(config.simulation_parameters.gamma_mean, ("array", "float"), const['deg2rad'])
    simulation_settings['parameters']['gamma_width'] = read_list(config.simulation_parameters.gamma_width, ("array", "float"), const['deg2rad'])
    simulation_settings['parameters']['ratio'] = read_list(config.simulation_parameters.ratio, ("array", "float"))
    simulation_settings['parameters']['j_mean'] = read_list(config.simulation_parameters.j_mean, ("array", "float"))
    simulation_settings['parameters']['j_width'] = read_list(config.simulation_parameters.j_width, ("array", "float"))
    return simulation_settings

def read_calculation_settings(config):
    calculation_settings = {}
    calculation_settings["integration_method"] = config.calculation_settings.integration_method
    if calculation_settings["integration_method"] == "monte_carlo":
        calculation_settings["mc_sample_size"] = int(config.calculation_settings.mc_sample_size)
        calculation_settings["grid_size"] = {}
    elif calculation_settings["integration_method"] == "uniform_grids":
        calculation_settings["mc_sample_size"] = 0
        calculation_settings["grid_size"] = {}
        calculation_settings["grid_size"]["powder_averaging"] = int(config.calculation_settings.grid_size.powder_averaging)
        calculation_settings["grid_size"]["distances"] = int(config.calculation_settings.grid_size.distances)
        calculation_settings["grid_size"]["spherical_angles"] = int(config.calculation_settings.grid_size.spherical_angles)
        calculation_settings["grid_size"]["rotations"] = int(config.calculation_settings.grid_size.rotations)    
    else:
        raise ValueError('Invalid integration method!')
        sys.exit(1)   
    calculation_settings["distributions"] = {}
    calculation_settings["r"] = config.calculation_settings.distributions.r
    calculation_settings["xi"] = config.calculation_settings.distributions.xi
    calculation_settings["phi"] = config.calculation_settings.distributions.phi
    calculation_settings["alpha"] = config.calculation_settings.distributions.alpha
    calculation_settings["beta"] = config.calculation_settings.distributions.beta
    calculation_settings["gamma"] = config.calculation_settings.distributions.gamma
    calculation_settings["j"] = config.calculation_settings.distributions.j   
    calculation_settings["excitation_treshold"] = float(config.calculation_settings.excitation_treshold)
    return calculation_settings

def read_output_settings(config):
    output_settings = {}
    output_settings['directory'] = config.output.directory
    output_settings['save_data'] = bool(config.output.save_data)
    output_settings['save_figures'] = bool(config.output.save_figures)
    return output_settings
  
def read_config(filepath):  
    sys.stdout.write('\n') 
    sys.stdout.write('Reading out the configuration file... ') 
    mode = {}
    experiments = []
    spins = []
    simulation_settings = {}
    calculation_settings = {}
    output_settings = {}
    with io.open(filepath) as file:
        config = libconf.load(file)
        mode = read_calculation_mode(config)
        experiments = read_experimental_parameters(config)
        spins = read_spin_parameters(config)
        if mode['simulation']:
            simulation_settings = read_simulation_settings(config)
        calculation_settings = read_calculation_settings(config)
        output_settings = read_output_settings(config)
    sys.stdout.write('[DONE]\n\n')  
    return [mode, experiments, spins, simulation_settings, calculation_settings, output_settings]
