import os
import sys
import io
import libconf
import numpy as np
sys.path.append('..')
from input.read_list import read_list
from input.read_tuple import read_tuple
from input.compare_size import compare_size
from input.parameter_object import ParameterObject
from input.parameter_id import ParameterID
from experiments.experiment_types import experiment_types
from spin_physics.spin import Spin
from background.background_types import background_types
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
    switch = int(config['mode'])
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
    ''' Reads out the parameters of PDS experiments and corresponding PDS data '''
    experiments = []
    for instance in config['experiments']:
        name = instance['name']
        technique = instance['technique'] 
        if technique in experiment_types:
            # Init experiment 
            experiment = experiment_types[technique](name)
            # Set the corresponding experimental parameters
            parameter_values = {}
            for parameter_name in experiment.parameter_names:
                if experiment.parameter_names[parameter_name] == 'float':
                    parameter_values[parameter_name] = float(instance[parameter_name])
                elif experiment.parameter_names[parameter_name] == 'int':
                    parameter_values[parameter_name] = int(instance[parameter_name]) 
                elif experiment.parameter_names[parameter_name] == 'str':
                    parameter_values[parameter_name] = int(instance[parameter_name])     
                elif experiment.parameter_names[parameter_name] == 'float_list':
                    parameter_values[parameter_name] = np.array(read_list(instance[parameter_name], parameter_name, float))
                elif experiment.parameter_names[parameter_name] == 'int_list':
                    parameter_values[parameter_name] = np.array(read_list(instance[parameter_name], parameter_name, int))
                elif experiment.parameter_names[parameter_name] == 'str_list':
                    parameter_values[parameter_name] = np.array(read_list(instance[parameter_name], parameter_name, str))    
                else:
                    raise ValueError('Unsupported format of \'{0}\'!'.format(parameter_name))
                    sys.exit(1)
            experiment.set_parameters(parameter_values)
            # Set the phase
            if 'phase' in instance:
                phase = float(instance['phase'])
            else:
                phase = np.nan
            # Set the zero point
            if 'zero_point' in instance:
                zero_point = float(instance['zero_point'])
            else:
                zero_point = np.nan
            # Set the noise level
            if 'noise_std' in instance:
                noise_std = float(instance['noise_std'])
            else:
                noise_std = np.nan
            # Set the corresponding PDS time trace
            experiment.signal_from_file(instance.filename, phase, zero_point, noise_std)
            if experiment.noise_std == 0:
                raise ValueError('Error: The zero level of noise is encountered!\n\
                Specify the nonzero quadrature componentof the PDS time trace or\n\
                provide the noise level explicitly via noise_std.')
                sys.exit(1)
            sys.stdout.write('\nExperiment \'{0}\' was loaded\n'.format(name))
            sys.stdout.write('Phase correction: {0:.0f} deg\n'.format(experiment.phase))
            sys.stdout.write('Zero point: {0:.3f} us\n'.format(experiment.zero_point))
            sys.stdout.write('Noise std: {0:0.6f}\n'.format(experiment.noise_std))
            sys.stdout.flush()
            # Add to the experiments list
            experiments.append(experiment)           
        else:
            raise ValueError('Invalid type of experiment!')
            sys.exit(1)  
    if experiments == []:
        raise ValueError('At least one experiment has to be provided!')
        sys.exit(1)
    return experiments
    

def read_spin_parameters(config):
    ''' Reads out the parameters of a spin system '''
    spins = []
    for instance in config['spins']:
        g = np.array(read_list(instance['g'], 'g', float))
        if g.size != 3:
            raise ValueError('Invalid number of elements in g!')
            sys.exit(1)
        gStrain = np.array(read_list(instance['gStrain'], 'gStrain', float))
        if gStrain.size != 0 and gStrain.size != 3:
            raise ValueError('Invalid number of elements in gStrain!')
            sys.exit(1)
        n = np.array(read_tuple(instance['n'], 'n', int))
        I = np.array(read_tuple(instance['I'], 'I', float))
        Abund = np.array(read_tuple(instance['Abund'], 'Abund', float))
        if I.size != n.size:
            raise ValueError('Number of elements in n\' and I\' must be equal!')
            sys.exit(1)
        if n.size != 0:
            A = np.array(read_tuple(instance['A'], 'A', float))
            if A.size != 3 * n.size:
                raise ValueError('Invalid number of elements in A!')
                sys.exit(1)
        else:
            A = np.array([])      
        if A.size != 0:
            AStrain = np.array(read_tuple(instance['AStrain'], 'AStrain', float))
            if AStrain.size != 0 and AStrain.size != A.size: 
                raise ValueError('Invalid number of elements in AStrain!')
                sys.exit(1)
        else:
            AStrain = np.array([]) 
        lwpp = float(instance['lwpp'])  
        T1 = float(instance['T1'])
        g_anisotropy_in_dipolar_coupling = bool(instance['g_anisotropy_in_dipolar_coupling'])
        spin = Spin(g, n, I, Abund, A, gStrain, AStrain, lwpp, T1, g_anisotropy_in_dipolar_coupling)
        spins.append(spin)
    if len(spins) != 2:
        raise ValueError('Unexpected number of spins!')
        sys.exit(1)
    return spins


def read_background_parameters(config):
    ''' Reads the parameters of the PDS background model '''
    background_model = config['background']['background_model']
    if background_model in background_types:
        background = (background_types[background_model])()
        background_parameters = {}
        for parameter_name in background.parameter_names:
            background_parameters[parameter_name] = {}
            background_parameters[parameter_name]['optimize'] = bool(config['background']['background_parameters'][parameter_name]['optimize'])
            if background_parameters[parameter_name]['optimize']:
                optimization_range = read_list(config['background']['background_parameters'][parameter_name]['range'], parameter_name, float)
                if len(optimization_range) == 2:
                    background_parameters[parameter_name]['range'] = optimization_range
                else:
                    raise ValueError('Invalid ranges of the background parameter \'{0}\'!'.format(parameter_name))
                    sys.exit(1)
            else:
                background_parameters[parameter_name]['range'] = []
            background_parameters[parameter_name]['value'] = float(config['background']['background_parameters'][parameter_name]['value'])
        background.set_parameters(background_parameters)
    else:   
        raise ValueError('Unsupported background model!')
        sys.exit(1)
    return background


def read_simulation_parameters(config):
    ''' Reads out the simulation parameters of the model '''
    simulation_parameters = {}
    for parameter_name in const['model_parameter_names']:
        simulation_parameters[parameter_name] = read_tuple(config['simulation_parameters'][parameter_name], parameter_name, float, const['model_parameter_scales'][parameter_name])
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
    ''' Reads out the fitting parameters of the model '''
    fitting_parameters = {}
    fitting_parameters['indices'] = {}
    fitting_parameters['ranges'] = []
    fitting_parameters['values'] = []
    no_fitting_parameter = 0
    no_fixed_parameter = 0  
    for parameter_name in const['model_parameter_names']:
        list_optimize = read_tuple(config['fitting_parameters'][parameter_name]['optimize'], parameter_name, int)
        list_range = read_tuple(config['fitting_parameters'][parameter_name]['range'], parameter_name, float, const['model_parameter_scales'][parameter_name])
        list_value = read_tuple(config['fitting_parameters'][parameter_name]['value'], parameter_name, float, const['model_parameter_scales'][parameter_name])
        parameter_objects = []
        index_range = 0
        index_value = 0
        for i in range(len(list_optimize)):
            if list_optimize[i] == 1:
                parameter_object = ParameterObject(list_optimize[i], no_fitting_parameter)
                parameter_objects.append(parameter_object)
                parameter_range = list_range[index_range]
                fitting_parameters['ranges'].append(parameter_range)
                index_range += 1
                no_fitting_parameter += 1
            elif list_optimize[i]  == 0:
                parameter_object = ParameterObject(list_optimize[i], no_fixed_parameter)
                parameter_objects.append(parameter_object)
                parameter_value = list_value[index_value]
                fitting_parameters['values'].append(parameter_value)
                index_value += 1
                no_fixed_parameter += 1
        fitting_parameters['indices'][parameter_name] = parameter_objects
    return fitting_parameters
    

def read_fitting_settings(config, experiments):
    ''' Reads out the fitting settings '''
    optimizer = None
    goodness_of_fit = config['fitting_settings']['goodness_of_fit']
    method = config['fitting_settings']['optimization_method']
    if method in optimization_methods and goodness_of_fit in const['goodness_of_fit_names']:
        optimizer = optimization_methods[method](method)
        optimizer.set_goodness_of_fit(goodness_of_fit)
        parameter_values = {}
        for parameter_name in optimizer.parameter_names:
            if optimizer.parameter_names[parameter_name] == 'float':
                parameter_values[parameter_name] = float(config['fitting_settings']['parameters'][parameter_name])
            elif optimizer.parameter_names[parameter_name] == 'int':
                parameter_values[parameter_name] = int(config['fitting_settings']['parameters'][parameter_name]) 
            elif optimizer.parameter_names[parameter_name] == 'str':
                parameter_values[parameter_name] = config['fitting_settings']['parameters'][parameter_name]   
            elif optimizer.parameter_names[parameter_name] == 'float_list':
                parameter_values[parameter_name] = np.array(read_list(config['fitting_settings']['parameters'][parameter_name], parameter_name, float))
            elif optimizer.parameter_names[parameter_name] == 'int_list':
                parameter_values[parameter_name] = np.array(read_list(config['fitting_settings']['parameters'][parameter_name], parameter_name, int)) 
            elif optimizer.parameter_names[parameter_name] == 'str_list':
                parameter_values[parameter_name] = np.array(read_list(config['fitting_settings']['parameters'][parameter_name], parameter_name, str))                 
            else:
                raise ValueError('Unsupported format of parameter \'{0}\'!'.format(parameter_name))
                sys.exit(1)
        optimizer.set_intrinsic_parameters(parameter_values)
    else:
        raise ValueError('Unsupported optimization method or/and goodness of fit parameter!')
        sys.exit(1)
    return optimizer


def read_error_analysis_parameters(config, fitting_parameters):
    ''' Reads out the parameters of the error analysis '''
    error_analysis_parameters = []
    parameter_names = read_tuple(config['error_analysis_parameters']['names'], 'error analysis parameter\'s names', str)
    if len(parameter_names) != 0:
        components = read_tuple(config['error_analysis_parameters']['components'], 'error analysis parameter\'s components', int)
        compare_size(parameter_names, components, 'error analysis parameter\'s names', 'error analysis parameter\'s components')
        for i in range(len(parameter_names)):
            subset_error_analysis_parameters = []
            for j in range(len(parameter_names[i])):
                parameter_name = parameter_names[i][j]
                component = components[i][j]-1
                parameter_id = ParameterID(parameter_name, component)
                subset_error_analysis_parameters.append(parameter_id)
            error_analysis_parameters.append(subset_error_analysis_parameters)
    # Check that the fitting and error analysis parameters are consistent with each other
    for i in range(len(error_analysis_parameters)):
        for j in range(len(error_analysis_parameters[i])):
            parameter_id = error_analysis_parameters[i][j]
            try:
                if parameter_id.is_optimized(fitting_parameters['indices']) == 0:
                    raise ValueError('All error analysis parameters must be the fitting parameters!')
                    sys.exit(1)  
            except IndexError:
                sys.stdout.write('All error analysis parameters must be the fitting parameters!\n')
                sys.stdout.flush()
                sys.exit(1)
    return error_analysis_parameters


def read_error_analysis_settings(config, mode):
    ''' Reads out the error analysis settings '''
    error_analyzer = ErrorAnalyzer()
    error_analysis_parameters = {}
    for parameter_name in error_analyzer.parameter_names:
        if error_analyzer.parameter_names[parameter_name] == 'float':
            error_analysis_parameters[parameter_name] = float(config['error_analysis_settings'][parameter_name])
        elif error_analyzer.parameter_names[parameter_name] == 'int':
            error_analysis_parameters[parameter_name] = int(config['error_analysis_settings'][parameter_name]) 
        elif error_analyzer.parameter_names[parameter_name] == 'str':
            error_analysis_parameters[parameter_name] = config['error_analysis_settings'][parameter_name]   
        elif error_analyzer.parameter_names[parameter_name] == 'float_list':
            error_analysis_parameters[parameter_name] = np.array(read_list(config['error_analysis_settings'][parameter_name], parameter_name, float))
        elif error_analyzer.parameter_names[parameter_name] == 'int_list':
            error_analysis_parameters[parameter_name] = np.array(read_list(config['error_analysis_settings'][parameter_name], parameter_name, int)) 
        elif error_analyzer.parameter_names[parameter_name] == 'str_list':
            error_analysis_parameters[parameter_name] = np.array(read_list(config['error_analysis_settings'][parameter_name], parameter_name, str))                 
        else:
            raise ValueError('Unsupported format of parameter \'{0}\'!'.format(parameter_name))
            sys.exit(1)
    error_analyzer.set_intrinsic_parameters(error_analysis_parameters)
    if not mode['fitting'] and mode['error_analysis']:
        if error_analysis_parameters['filepath_optimized_parameters'] == '':
            raise ValueError('A file with optimized fitting parameters needs to be provided!')
            sys.exit(1)
    return error_analyzer


def read_calculation_settings(config):
    ''' Reads out the calculation settings '''  
    integration_method = config['calculation_settings']['integration_method']
    if integration_method in simulator_types:
        simulator = (simulator_types[integration_method])()
        calculation_settings = {}
        for parameter_name in simulator.parameter_names:
            if simulator.parameter_names[parameter_name] == 'float':
                calculation_settings[parameter_name] = float(config['calculation_settings'][parameter_name])
            elif simulator.parameter_names[parameter_name] == 'int':
                calculation_settings[parameter_name] = int(config['calculation_settings'][parameter_name]) 
            elif simulator.parameter_names[parameter_name] == 'str':
                calculation_settings[parameter_name] = config['calculation_settings'][parameter_name]   
            elif simulator.parameter_names[parameter_name] == 'float_list':
                calculation_settings[parameter_name] = np.array(read_list(config['calculation_settings'][parameter_name], parameter_name, float))
            elif simulator.parameter_names[parameter_name] == 'int_list':
                calculation_settings[parameter_name] = np.array(read_list(config['calculation_settings'][parameter_name], parameter_name, int))    
            elif simulator.parameter_names[parameter_name] == 'str_list':
                calculation_settings[parameter_name] = np.array(read_list(config['calculation_settings'][parameter_name], parameter_name, str))
            else:
                raise ValueError('Unsupported format of \'{0}\'!'.format(parameter_name))
                sys.exit(1) 
        distribution_types = {}
        for name in ['r', 'xi', 'phi', 'alpha', 'beta', 'gamma', 'j']:
            distribution_types[name] = config['calculation_settings']['distribution_types'][name]
        for key in distribution_types:
            if not distribution_types[key] in const['distribution_types']:
                raise ValueError('Unsupported type of distribution is encountered for %s!' % (key))
                sys.exit(1)
        calculation_settings['distribution_types'] = distribution_types                
        calculation_settings['excitation_threshold'] = float(config['calculation_settings']['excitation_threshold'])
        calculation_settings['euler_angles_convention'] = config['calculation_settings']['euler_angles_convention']
        if not calculation_settings['euler_angles_convention'] in const['euler_angles_conventions']:
            raise ValueError('Unsupported Euler angles convention!')
            sys.exit(1)
        simulator.set_calculation_settings(calculation_settings)
    return simulator

  
def read_config(filepath): 
    ''' Reads input data from a configuration file '''
    sys.stdout.write('\n########################################################################\
                      \n# Reading the configuration file and preprocessing the PDS time traces #\
                      \n########################################################################\n')
    sys.stdout.flush()
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
        background = read_background_parameters(config)
        if mode['simulation']:
            simulation_parameters = read_simulation_parameters(config)
        elif mode['fitting'] or mode['error_analysis']:
            fitting_parameters = read_fitting_parameters(config)
            optimizer = read_fitting_settings(config, experiments)
            error_analysis_parameters = read_error_analysis_parameters(config, fitting_parameters)
            error_analyzer = read_error_analysis_settings(config, mode)
        simulator = read_calculation_settings(config)
        plotter = Plotter(data_saver)
    return mode, experiments, spins, background, simulator, simulation_parameters, optimizer, fitting_parameters, \
        error_analyzer, error_analysis_parameters, data_saver, plotter