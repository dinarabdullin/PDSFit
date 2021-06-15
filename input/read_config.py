import os
import sys
import io
import libconf
import numpy as np
sys.path.append('..')
from input.read_tuple import read_tuple
from input.read_list import read_list
from input.compare_size import compare_size
from input.read_parameter import read_parameter
from input.parameter_object import ParameterObject
from input.parameter_id import ParameterID
from experiments.experiment_types import experiment_types
from spin_physics.spin import Spin
from simulation.simulator_types import simulator_types
from fitting.optimization_methods import optimization_methods
from error_analysis.error_analyzer import ErrorAnalyzer
from output.data_saver import DataSaver
from output.logger import Logger
from plots.plotter import Plotter
from supplement.definitions import const


def read_output_settings(config, filepath_config):
    ''' Reads out the output settings '''
    save_data = bool(config.output.save_data)
    save_figures = bool(config.output.save_figures)
    output_directory = config.output.directory
    data_saver = DataSaver(save_data, save_figures)
    data_saver.create_output_directory(output_directory, filepath_config)
    if data_saver.directory != '':
        sys.stdout = Logger(data_saver.directory+'logfile.log')
    return data_saver


def read_calculation_mode(config):
    ''' Reads out the calculation mode '''
    mode = {}
    switch = int(config.mode)
    if switch == 0:
        mode['simulation'] = 1
        mode['fitting'] = 0
        mode['error_analysis'] = 0
    elif switch == 1:
        mode['simulation'] = 0
        mode['fitting'] = 1
        mode['error_analysis'] = 1
    elif switch == 2:
        mode['simulation'] = 0
        mode['fitting'] = 0
        mode['error_analysis'] = 1
    else:
        raise ValueError('Invalid mode!')
        sys.exit(1)
    return mode


def read_experimental_parameters(config, mode):
    ''' Reads out the experimental parameters '''
    experiments = []
    list_noise_std = []
    for instance in config.experiments:
        name = instance.name
        technique = instance.technique 
        if technique in experiment_types:
            experiment = experiment_types[technique](name)
            experiment.signal_from_file(instance.filename)
            noise_std = float(instance.noise_std)
            if noise_std:
                experiment.set_noise_std(noise_std)
            list_noise_std.append(experiment.noise_std)
            magnetic_field = float(instance.magnetic_field)
            detection_frequency = float(instance.detection_frequency)
            detection_pulse_lengths = []
            for pulse_length in instance.detection_pulse_lengths:
                detection_pulse_lengths.append(float(pulse_length))
            if experiment.technique == 'peldor':
                pump_frequency = float(instance.pump_frequency)
                pump_pulse_lengths = []
                for pulse_length in instance.pump_pulse_lengths:
                    pump_pulse_lengths.append(float(pulse_length))
                experiment.set_parameters(magnetic_field, detection_frequency, detection_pulse_lengths, pump_frequency, pump_pulse_lengths)
            elif experiment.technique == 'ridme':
                mixing_time = float(instance.mixing_time)
                temperature = float(instance.temperature)
                experiment.set_parameters(magnetic_field, detection_frequency, detection_pulse_lengths, mixing_time, temperature)  
            else:
                raise ValueError('Unsupported technique!')
                sys.exit(1)
            experiments.append(experiment)
            print('\nExperiment \'{0}\' was loaded'.format(name))
            print('Phase correction: {0:.0f} deg'.format(experiment.phase))
            print('Zero point: {0:.3f} us'.format(experiment.zero_point))
            print('Noise std: {0:0.6f}'.format(experiment.noise_std))
        else:
            raise ValueError('Invalid type of experiment!')
            sys.exit(1)  
    if experiments == []:
        raise ValueError('At least one experiment has to be provided!')
        sys.exit(1)
    list_noise_std = np.array(list_noise_std)
    indices_zero_values = np.where(list_noise_std==0)[0]
    if mode['simulation'] or mode['fitting']:
        if (indices_zero_values.size != 0):
            for experiment in experiments:
                experiment.set_noise_std(1)
            print('\nZero is encountered among the standard deviations of noise!')
            print('To avoid problems with the scoring, the standard deviation of noise is set to 1 for all experiments.')
    if mode['error_analysis']: 
        if indices_zero_values.size != 0:
            mode['error_analysis'] = 0
            print('Error analysis is disabled! To enable error analysis, provide non-zero standard deviations of noise for all experiments.')
    return experiments


def read_spin_parameters(config):
    ''' Reads out the spin system parameters '''
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
            raise ValueError('Number of elements in n\' and I\' must be equal!')
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
    ''' Reads out the simulation parameters '''
    simulation_parameters = {}
    simulation_parameters = {}
    simulation_parameters['r_mean'] = read_parameter(config.simulation_parameters.r_mean, 'r_mean', 'float')
    simulation_parameters['r_width'] = read_parameter(config.simulation_parameters.r_width, 'r_width', 'float')      
    simulation_parameters['xi_mean'] = read_parameter(config.simulation_parameters.xi_mean, 'xi_mean', 'float', const['deg2rad'])
    simulation_parameters['xi_width'] = read_parameter(config.simulation_parameters.xi_width, 'xi_width', 'float', const['deg2rad'])
    simulation_parameters['phi_mean'] = read_parameter(config.simulation_parameters.phi_mean, 'phi_mean', 'float', const['deg2rad'])
    simulation_parameters['phi_width'] = read_parameter(config.simulation_parameters.phi_width, 'phi_width', 'float', const['deg2rad'])
    simulation_parameters['alpha_mean'] = read_parameter(config.simulation_parameters.alpha_mean, 'alpha_mean', 'float', const['deg2rad'])
    simulation_parameters['alpha_width'] = read_parameter(config.simulation_parameters.alpha_width, 'alpha_width', 'float', const['deg2rad'])
    simulation_parameters['beta_mean'] = read_parameter(config.simulation_parameters.beta_mean, 'beta_mean', 'float', const['deg2rad'])
    simulation_parameters['beta_width'] = read_parameter(config.simulation_parameters.beta_width, 'beta_width', 'float', const['deg2rad'])
    simulation_parameters['gamma_mean'] = read_parameter(config.simulation_parameters.gamma_mean, 'gamma_mean', 'float', const['deg2rad'])
    simulation_parameters['gamma_width'] = read_parameter(config.simulation_parameters.gamma_width, 'gamma_width', 'float', const['deg2rad'])
    simulation_parameters['rel_prob'] = read_parameter(config.simulation_parameters.rel_prob, 'rel_prob', 'float')
    simulation_parameters['j_mean'] = read_parameter(config.simulation_parameters.j_mean, 'j_mean', 'float')
    simulation_parameters['j_width'] = read_parameter(config.simulation_parameters.j_width, 'j_width', 'float')
    # Compare the sizes of the related parameters
    compare_size(simulation_parameters['r_mean'], simulation_parameters['r_width'], 'r_mean', 'r_width')
    compare_size(simulation_parameters['xi_mean'], simulation_parameters['xi_width'], 'xi_mean', 'xi_width')
    compare_size(simulation_parameters['phi_mean'], simulation_parameters['phi_width'], 'phi_mean', 'phi_width')
    compare_size(simulation_parameters['alpha_mean'], simulation_parameters['alpha_width'],'alpha_mean', 'alpha_width')
    compare_size(simulation_parameters['beta_mean'], simulation_parameters['beta_width'], 'beta_mean', 'beta_width')
    compare_size(simulation_parameters['gamma_mean'], simulation_parameters['gamma_width'], 'gamma_mean', 'gamma_width')
    compare_size(simulation_parameters['j_mean'], simulation_parameters['j_width'], 'j_mean', 'j_width')      
    return simulation_parameters


def read_fitting_parameters(config):
    ''' Reads out the fitting parameters '''
    fitting_parameters = {}
    fitting_parameters['indices'] = {}
    fitting_parameters['ranges'] = []
    fitting_parameters['values'] = []
    no_fitting_parameter = 0
    no_fixed_parameter = 0  
    for parameter in const['fitting_parameters_names']:
        list_optimize = read_parameter(config.fitting_parameters[parameter]['optimize'], parameter, 'int')
        list_range = read_tuple(config.fitting_parameters[parameter]['range'], ('array','float'), const['fitting_parameters_scales'][parameter])
        list_value = read_tuple(config.fitting_parameters[parameter]['value'], ('float',), const['fitting_parameters_scales'][parameter])
        list_parameter_objects = []
        index_range = 0
        index_value = 0
        for i in range(len(list_optimize)):
            sublist_parameter_objects = []
            for j in range(len(list_optimize[i])):
                if list_optimize[i][j] == 1:
                    parameter_object = ParameterObject(list_optimize[i][j], no_fitting_parameter)
                    sublist_parameter_objects.append(parameter_object)
                    parameter_range = list_range[index_range]
                    fitting_parameters['ranges'].append(parameter_range)
                    index_range += 1
                    no_fitting_parameter += 1
                elif list_optimize[i][j]  == 0:
                    parameter_object = ParameterObject(list_optimize[i][j], no_fixed_parameter)
                    sublist_parameter_objects.append(parameter_object)
                    parameter_value = list_value[index_value]
                    fitting_parameters['values'].append(parameter_value)
                    index_value += 1
                    no_fixed_parameter += 1
            list_parameter_objects.append(sublist_parameter_objects)
        fitting_parameters['indices'][parameter] = list_parameter_objects
    return fitting_parameters
    

def read_fitting_settings(config, experiments):
    ''' Reads out the fitting settings '''
    optimizer = None
    method = config.fitting_settings.optimization_method
    display_graphics = int(config.fitting_settings.display_graphics)
    goodness_of_fit = config.fitting_settings.goodness_of_fit
    if (goodness_of_fit == 'chi2') or (goodness_of_fit == 'reduced_chi2'):
        list_noise_std = []
        for experiment in experiments:
            list_noise_std.append(experiment.noise_std)
        indices_zero_values = np.where(list_noise_std==0)[0]
        if indices_zero_values.size != 0:
            goodness_of_fit = 'chi2_noise_std_1'
    if method in optimization_methods and goodness_of_fit in const['goodness_of_fit_names']:
        optimizer = optimization_methods[method](method, display_graphics, goodness_of_fit)
        if optimizer.name == 'ga':
            number_of_generations = int(config.fitting_settings.ga_parameters.number_of_generations)
            generation_size = int(config.fitting_settings.ga_parameters.generation_size)
            crossover_probability = float(config.fitting_settings.ga_parameters.crossover_probability)
            mutation_probability = float(config.fitting_settings.ga_parameters.mutation_probability)
            parent_selection = str(config.fitting_settings.ga_parameters.parent_selection)
            optimizer.set_intrinsic_parameters(number_of_generations, generation_size, crossover_probability, mutation_probability, parent_selection)   
    else:
        raise ValueError('Unsupported optimization method or/and goodness-of-fit parameter!')
        sys.exit(1)
    return optimizer


def read_error_analysis_parameters(config, fitting_parameters):
    ''' Reads out the error analysis parameters '''
    error_analysis_parameters = []
    parameters = read_tuple(config.error_analysis_parameters.parameters, ('array','str'))
    if len(parameters) != 0:
        spin_pairs = read_tuple(config.error_analysis_parameters.spin_pairs, ('array','int'))
        if len(spin_pairs) != 0:
            compare_size(parameters, spin_pairs, 'parameters', 'spin_pairs')
        components = read_tuple(config.error_analysis_parameters.components, ('array','int'))
        if len(components) != 0:
            compare_size(parameters, components, 'parameters', 'components')
        for i in range(len(parameters)):
            list_parameter_id = []
            for j in range(len(parameters[i])):
                parameter = parameters[i][j]
                if len(spin_pairs) != 0:
                    spin_pair = spin_pairs[i][j]-1
                else:
                    spin_pair = 0
                if len(components) != 0:
                    component = components[i][j]-1
                else:
                    component = 0
                parameter_id = ParameterID(parameter, spin_pair, component)
                list_parameter_id.append(parameter_id)
            error_analysis_parameters.append(list_parameter_id)
    # Check that the fitting and error analysis parameters are consistent with each other
    for i in range(len(error_analysis_parameters)):
        for j in range(len(error_analysis_parameters[i])):
            parameter_id = error_analysis_parameters[i][j]
            try:
                if parameter_id.is_optimized(fitting_parameters['indices']) == 0:
                    raise ValueError('The parameters of error analysis must be in the fitting parameters list!')
                    sys.exit(1)  
            except IndexError:
                print('At least one parameter of error analysis is absent in the fitting parameters list!')
                sys.exit(1)
    return error_analysis_parameters


def read_error_analysis_settings(config, mode):
    ''' Reads out the error analysis settings '''
    error_analysis_parameters = {}
    error_analysis_parameters['sample_size'] = int(config.error_analysis_settings.sample_size)
    error_analysis_parameters['confidence_interval'] = float(config.error_analysis_settings.confidence_interval)
    error_analysis_parameters['filepath_optimized_parameters'] = ''
    if not mode['fitting'] and mode['error_analysis']:
        error_analysis_parameters['filepath_optimized_parameters'] = config.error_analysis_settings.filepath_optimized_parameters
        if error_analysis_parameters['filepath_optimized_parameters'] == '':
            raise ValueError('A file with the optimized fitting parameters has to be provided!')
            sys.exit(1)
    error_analyzer = ErrorAnalyzer(error_analysis_parameters)
    return error_analyzer


def read_calculation_settings(config):
    ''' Reads out the calculation settings '''
    calculation_settings = {}
    integration_method = config.calculation_settings.integration_method
    if integration_method in simulator_types:
        if integration_method == 'monte_carlo':
            calculation_settings['mc_sample_size'] = int(config.calculation_settings.mc_sample_size)
        elif integration_method == 'grids':
            grid_size = {}
            grid_size['powder_averaging'] = int(config.calculation_settings.grid_size.powder_averaging)
            grid_size['distances'] = int(config.calculation_settings.grid_size.distances)
            grid_size['spherical_angles'] = int(config.calculation_settings.grid_size.spherical_angles)
            grid_size['rotations'] = int(config.calculation_settings.grid_size.rotations) 
            calculation_settings['grid_size'] = grid_size
        distributions = {}
        distributions['r'] = config.calculation_settings.distributions.r
        distributions['xi'] = config.calculation_settings.distributions.xi
        distributions['phi'] = config.calculation_settings.distributions.phi
        distributions['alpha'] = config.calculation_settings.distributions.alpha
        distributions['beta'] = config.calculation_settings.distributions.beta
        distributions['gamma'] = config.calculation_settings.distributions.gamma
        distributions['j'] = config.calculation_settings.distributions.j
        for key in distributions:
            if not distributions[key] in const['distribution_types']:
                raise ValueError('Unsupported type of distribution for %s' % (key))
                sys.exit(1)
        calculation_settings['distributions'] = distributions                
        calculation_settings['excitation_treshold'] = float(config.calculation_settings.excitation_treshold)
        calculation_settings['euler_angles_convention'] = config.calculation_settings.euler_angles_convention
        if not calculation_settings['euler_angles_convention'] in const['euler_angles_conventions']:
            raise ValueError('Unsupported Euler angles convention')
            sys.exit(1)
        background = {'decay_constant': {}, 'dimension': {}, 'scale_factor': {}}
        background['decay_constant']['optimize'] = bool(config.calculation_settings.background_parameters.decay_constant)
        background['dimension']['optimize'] = bool(config.calculation_settings.background_parameters.dimension)
        background['scale_factor']['optimize'] = bool(config.calculation_settings.background_parameters.scale_factor)
        background['decay_constant']['value'] = float(config.calculation_settings.background_parameter_values.decay_constant)
        background['dimension']['value'] = float(config.calculation_settings.background_parameter_values.dimension)
        background['scale_factor']['value'] = float(config.calculation_settings.background_parameter_values.scale_factor)
        if background['decay_constant']['optimize']:
            background['decay_constant']['ranges'] = read_list(config.calculation_settings.background_parameter_ranges.decay_constant, 'float')
            if len(background['decay_constant']['ranges']) != 2:
                raise ValueError('Invalid ranges for a background parameter!')
                sys.exit(1)
        else:
            background['decay_constant']['ranges'] = []
        if background['dimension']['optimize']:
            background['dimension']['ranges'] = read_list(config.calculation_settings.background_parameter_ranges.dimension, 'float')
            if len(background['dimension']['ranges']) != 2:
                raise ValueError('Invalid ranges for a background parameter!')
                sys.exit(1)
        else:
            background['dimension']['ranges'] = []
        if background['scale_factor']['optimize']:
            background['scale_factor']['ranges'] = read_list(config.calculation_settings.background_parameter_ranges.decay_constant, 'float')
            if len(background['scale_factor']['ranges']) != 2:
                raise ValueError('Invalid ranges for a background parameter!')
                sys.exit(1)
        else:
            background['scale_factor']['ranges'] = []
        calculation_settings['background'] = background
        simulator = (simulator_types[integration_method])(calculation_settings)
    return simulator

  
def read_config(filepath): 
    ''' Reads input data from a configuration file '''
    print('\nReading out the configuration file...') 
    simulation_parameters = {}
    fitting_parameters = {}
    optimizer = None
    error_analysis_parameters = []
    error_analyzer = None
    with io.open(filepath) as file:
        config = libconf.load(file)
        data_saver = read_output_settings(config, filepath) 
        mode = read_calculation_mode(config)
        experiments = read_experimental_parameters(config, mode)
        spins = read_spin_parameters(config)
        if mode['simulation']:
            simulation_parameters = read_simulation_parameters(config)
        elif mode['fitting'] or mode['error_analysis']:
            fitting_parameters = read_fitting_parameters(config)
            optimizer = read_fitting_settings(config, experiments)
            error_analysis_parameters = read_error_analysis_parameters(config, fitting_parameters)
            error_analyzer = read_error_analysis_settings(config, mode)
        simulator = read_calculation_settings(config)
        plotter = Plotter(data_saver)
    return mode, experiments, spins, simulation_parameters, fitting_parameters, optimizer, error_analysis_parameters, error_analyzer, simulator, data_saver, plotter