import io
import sys
import libconf
import numpy as np
sys.path.append('..')
from input.read_tuple import read_tuple
from input.read_list import read_list
from input.read_parameter import read_parameter
from input.validate_simulation_parameters import validate_simulation_parameters
from input.parameter_id import ParameterID
from experiments.experiment_types import experiment_types
from spin_physics.spin import Spin
from fitting.optimization_methods import optimization_methods
from supplement.definitions import const


def read_calculation_mode(config):
    ''' Read out the calculation mode '''
    mode = {}
    switch = int(config.mode)
    if switch == 0:
        mode['simulation'] = 1
        mode['fitting'] = 0
        mode['error_analysis'] = 0
    elif switch == 1:
        mode['simulation'] = 0
        mode['fitting'] = 1
        mode['error_analysis'] = 0
    elif switch == 2:
        mode['simulation'] = 0
        mode['fitting'] = 0
        mode['error_analysis'] = 1
    else:
        raise ValueError('Invalid mode!')
        sys.exit(1)
    return mode


def read_experimental_parameters(config):
    ''' Read out the experimental parameters '''
    experiments = []
    for instance in config.experiments:
        name = instance.name
        technique = instance.technique 
        if technique in experiment_types:
            exp = experiment_types[technique](name)
            exp.signal_from_file(instance.filename, 1)
            noise_std = float(instance.noise_std)
            exp.set_noise_std(noise_std)
            magnetic_field = float(instance.magnetic_field)
            detection_frequency = float(instance.detection_frequency)
            detection_pulse_lengths = []
            for pulse_length in instance.detection_pulse_lengths:
                detection_pulse_lengths.append(float(pulse_length))
            if exp.technique == 'peldor':
                pump_frequency = float(instance.pump_frequency)
                pump_pulse_lengths = []
                for pulse_length in instance.pump_pulse_lengths:
                    pump_pulse_lengths.append(float(pulse_length))
                exp.set_parameters(magnetic_field, detection_frequency, detection_pulse_lengths, pump_frequency, pump_pulse_lengths)
            elif exp.technique == 'ridme':
                mixing_time = float(instance.mixing_time)
                temperature = float(instance.temperature)
                exp.set_parameters(magnetic_field, detection_frequency, detection_pulse_lengths, mixing_time, temperature)  
            else:
                raise ValueError('Unsupported technique!')
                sys.exit(1)
            experiments.append(exp)
        else:
            raise ValueError('Invalid type of experiment!')
            sys.exit(1)  
    if experiments == []:
        raise ValueError('At least one experiment has to be provided!')
        sys.exit(1)
    return experiments

    
def read_spin_parameters(config):
    ''' Read out the spin system parameters '''
    spins = []
    for instance in config.spins:
        g = np.array(read_list(instance.g, 'float'))
        if g.size != 3:
            raise ValueError('Invalid number of elements in g!')
            sys.exit(1)
        gStrain = np.array(read_list(instance.gStrain, 'float'))
        if gStrain.size != 0 and gStrain.size != 3:
            raise ValueError('Invalid number of elements in gStrain!')
            sys.exit(1)
        n = np.array(read_tuple(instance.n, ('int',)))
        I = np.array(read_tuple(instance.I, ('float',)))
        if I.size != n.size:
            raise ValueError('Number of elements in n and I must be equal!')
            sys.exit(1)
        if n.size != 0:
            A = np.array(read_tuple(instance.A, ('array','float')))
            if A.size != 3 * n.size:
                raise ValueError('Invalid number of elements in A!')
                sys.exit(1)
        else:
            A = np.array([])      
        if A.size != 0:
            AStrain = np.array(read_tuple(instance.AStrain, ('array','float')))
            if AStrain.size != 0 and AStrain.size != A.size: 
                raise ValueError('Invalid number of elements in AStrain!')
                sys.exit(1)
        else:
            AStrain = np.array([]) 
        lwpp = float(instance.lwpp)  
        T1 = float(instance.T1)
        g_anisotropy_in_dipolar_coupling = bool(instance.g_anisotropy_in_dipolar_coupling)
        spin = Spin(g, n, I, A, gStrain, AStrain, lwpp, T1, g_anisotropy_in_dipolar_coupling)
        spins.append(spin)
    if len(spins) < 2:
        raise ValueError('Number of spins has to be larger than 2!')
        sys.exit(1)
    return spins


def read_simulation_parameters(config):
    ''' Read out the simulation parameters '''
    simulation_parameters = {}
    simulation_parameters = {}
    simulation_parameters['r_mean'] = read_parameter(config.simulation_parameters.r_mean, 'float')
    simulation_parameters['r_width'] = read_parameter(config.simulation_parameters.r_width, 'float')      
    simulation_parameters['xi_mean'] = read_parameter(config.simulation_parameters.xi_mean, 'float', const['deg2rad'])
    simulation_parameters['xi_width'] = read_parameter(config.simulation_parameters.xi_width, 'float', const['deg2rad'])
    simulation_parameters['phi_mean'] = read_parameter(config.simulation_parameters.phi_mean, 'float', const['deg2rad'])
    simulation_parameters['phi_width'] = read_parameter(config.simulation_parameters.phi_width, 'float', const['deg2rad'])
    simulation_parameters['alpha_mean'] = read_parameter(config.simulation_parameters.alpha_mean, 'float', const['deg2rad'])
    simulation_parameters['alpha_width'] = read_parameter(config.simulation_parameters.alpha_width, 'float', const['deg2rad'])
    simulation_parameters['beta_mean'] = read_parameter(config.simulation_parameters.beta_mean, 'float', const['deg2rad'])
    simulation_parameters['beta_width'] = read_parameter(config.simulation_parameters.beta_width, 'float', const['deg2rad'])
    simulation_parameters['gamma_mean'] = read_parameter(config.simulation_parameters.gamma_mean, 'float', const['deg2rad'])
    simulation_parameters['gamma_width'] = read_parameter(config.simulation_parameters.gamma_width, 'float', const['deg2rad'])
    simulation_parameters['rel_prob'] = read_parameter(config.simulation_parameters.rel_prob, 'float')
    simulation_parameters['j_mean'] = read_parameter(config.simulation_parameters.j_mean, 'float')
    simulation_parameters['j_width'] = read_parameter(config.simulation_parameters.j_width, 'float')
    validate_simulation_parameters(simulation_parameters, 'r_mean', 'r_width')
    validate_simulation_parameters(simulation_parameters, 'xi_mean', 'xi_width')
    validate_simulation_parameters(simulation_parameters, 'phi_mean', 'phi_width')
    validate_simulation_parameters(simulation_parameters, 'alpha_mean', 'alpha_width')
    validate_simulation_parameters(simulation_parameters, 'beta_mean', 'beta_width')
    validate_simulation_parameters(simulation_parameters, 'gamma_mean', 'gamma_width')
    validate_simulation_parameters(simulation_parameters, 'rel_prob')
    validate_simulation_parameters(simulation_parameters, 'j_mean', 'j_width')
    return simulation_parameters


def read_fitting_parameters(config):
    ''' Read out the fitting parameters '''
    fitting_parameters = {}
    fitting_parameters['indices'] = {}
    fitting_parameters['ranges'] = []
    fitting_parameters['values'] = []
    no_fitting_parameter = 0
    no_fixed_parameter = 0  
    for parameter in const['fitting_parameters_names']:
        optimize_list = read_parameter(config.fitting_parameters[parameter]['optimize'], 'int')
        range_list = read_tuple(config.fitting_parameters[parameter]['range'], ('array','float'), const['fitting_parameters_scales'][parameter])
        value_list = read_tuple(config.fitting_parameters[parameter]['value'], ('float',), const['fitting_parameters_scales'][parameter])
        parameter_id_list = []
        range_index = 0
        value_index = 0
        for i in range(len(optimize_list)):
            parameter_id_sublist = []
            for j in range(len(optimize_list[i])):
                if optimize_list[i][j] == 1:
                    parameter_id = ParameterID(optimize_list[i][j], no_fitting_parameter)
                    parameter_id_sublist.append(parameter_id)
                    parameter_range = range_list[range_index]
                    fitting_parameters['ranges'].append(parameter_range)
                    range_index += 1
                    no_fitting_parameter += 1
                elif optimize_list[i][j]  == 0:
                    parameter_id = ParameterID(optimize_list[i][j], no_fixed_parameter)
                    parameter_id_sublist.append(parameter_id)
                    parameter_value = value_list[value_index]
                    fitting_parameters['values'].append(parameter_value)
                    value_index += 1
                    no_fixed_parameter += 1
            parameter_id_list.append(parameter_id_sublist)
        fitting_parameters['indices'][parameter] = parameter_id_list
    return fitting_parameters
    

def read_fitting_settings(config):
    ''' Read out the fitting settings '''
    method = config.fitting_settings.optimization_method
    display_graphics = int(config.fitting_settings.display_graphics)
    if method in optimization_methods:
        optimizer = optimization_methods[method](method, display_graphics)
        if optimizer.name == 'ga':
            number_of_generations = int(config.fitting_settings.ga_parameters.number_of_generations)
            generation_size = int(config.fitting_settings.ga_parameters.generation_size)
            crossover_probability = float(config.fitting_settings.ga_parameters.crossover_probability)
            mutation_probability = float(config.fitting_settings.ga_parameters.mutation_probability)
            parent_selection = str(config.fitting_settings.ga_parameters.parent_selection)
            optimizer.set_intrinsic_parameters(number_of_generations, generation_size, crossover_probability,
                                               mutation_probability, parent_selection)   
    else:
        raise ValueError('Unsupported optimization method!')
        sys.exit(1)
    return optimizer


def read_calculation_settings(config):
    ''' Read out the calculation settings '''
    calculation_settings = {}
    calculation_settings['integration_method'] = config.calculation_settings.integration_method
    if calculation_settings['integration_method'] == 'monte_carlo':
        calculation_settings['mc_sample_size'] = int(config.calculation_settings.mc_sample_size)
        calculation_settings['grid_size'] = {}
    elif calculation_settings['integration_method'] == 'grids':
        calculation_settings['mc_sample_size'] = 0
        calculation_settings['grid_size'] = {}
        calculation_settings['grid_size']['powder_averaging'] = int(config.calculation_settings.grid_size.powder_averaging)
        calculation_settings['grid_size']['distances'] = int(config.calculation_settings.grid_size.distances)
        calculation_settings['grid_size']['spherical_angles'] = int(config.calculation_settings.grid_size.spherical_angles)
        calculation_settings['grid_size']['rotations'] = int(config.calculation_settings.grid_size.rotations)    
    else:
        raise ValueError('Invalid integration method!')
        sys.exit(1)   
    calculation_settings['distributions'] = {}
    calculation_settings['distributions']['r'] = config.calculation_settings.distributions.r
    calculation_settings['distributions']['xi'] = config.calculation_settings.distributions.xi
    calculation_settings['distributions']['phi'] = config.calculation_settings.distributions.phi
    calculation_settings['distributions']['alpha'] = config.calculation_settings.distributions.alpha
    calculation_settings['distributions']['beta'] = config.calculation_settings.distributions.beta
    calculation_settings['distributions']['gamma'] = config.calculation_settings.distributions.gamma
    calculation_settings['distributions']['j'] = config.calculation_settings.distributions.j
    for key in calculation_settings['distributions']:
        if not calculation_settings['distributions'][key] in const['distribution_types']:
            raise ValueError('Unsupported type of distribution for %s' % (key))
            sys.exit(1)
    calculation_settings['excitation_treshold'] = float(config.calculation_settings.excitation_treshold)
    calculation_settings['euler_angles_convention'] = config.calculation_settings.euler_angles_convention
    if not calculation_settings['euler_angles_convention'] in const['euler_angles_conventions']:
        raise ValueError('Unsupported Euler angles convention')
        sys.exit(1)
    calculation_settings['fit_modulation_depth'] = bool(config.calculation_settings.fit_modulation_depth)
    if calculation_settings['fit_modulation_depth']:
        calculation_settings['interval_modulation_depth'] = float(config.calculation_settings.interval_modulation_depth)
        calculation_settings['scale_range_modulation_depth'] = read_list(config.calculation_settings.scale_range_modulation_depth, 'float')
        if (len(calculation_settings['scale_range_modulation_depth']) != 0) and (len(calculation_settings['scale_range_modulation_depth']) != 2):
            raise ValueError('Invalid format of scale_range_modulation_depth!')
            sys.exit(1)
    calculation_settings['scale_chi2_by_modulation_depth'] = int(config.calculation_settings.scale_chi2_by_modulation_depth)
    return calculation_settings


def read_output_settings(config):
    ''' Read out the output settings '''
    output_settings = {}
    output_settings['directory'] = config.output.directory
    output_settings['save_data'] = bool(config.output.save_data)
    output_settings['save_figures'] = bool(config.output.save_figures)
    return output_settings

  
def read_config(filepath): 
    ''' Read input data from a configuration file '''
    print('\nReading out the configuration file...') 
    mode = {}
    experiments = []
    spins = []
    simulation_parameters = {}
    fitting_parameters = {}
    optimizer = None
    error_analysis_settings = {}
    calculation_settings = {}
    output_settings = {}
    with io.open(filepath) as file:
        config = libconf.load(file)
        mode = read_calculation_mode(config)
        experiments = read_experimental_parameters(config)
        spins = read_spin_parameters(config)
        if mode['simulation']:
            simulation_parameters = read_simulation_parameters(config)
        elif mode['fitting']:
            fitting_parameters = read_fitting_parameters(config)
            optimizer = read_fitting_settings(config)
        #elif mode['error_analysis']:
        #   ...
        calculation_settings = read_calculation_settings(config)
        if calculation_settings['fit_modulation_depth']:
            for experiment in experiments:
                experiment.compute_modulation_depth(calculation_settings['interval_modulation_depth'])
        output_settings = read_output_settings(config) 
    return mode, experiments, spins, simulation_parameters, fitting_parameters, optimizer, error_analysis_settings, calculation_settings, output_settings