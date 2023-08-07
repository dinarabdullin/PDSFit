import os
import io
import sys
import libconf
import wx
import numpy as np
from textwrap import wrap
import sys 
sys.path.append('..')
from supplement.definitions import const
from input.read_config import read_fitting_parameters, read_error_analysis_settings
from input.load_optimized_model_parameters import load_optimized_model_parameters
from input.parameter_id import ParameterID
from output.data_saver import DataSaver
from output.logger import Logger
from plots.plotter import Plotter
from output.fitting.print_model_parameters import print_model_parameters


def get_filepath(message):
    app = wx.App(None) 
    dialog = wx.FileDialog(None, message, wildcard='*.*', style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
    if dialog.ShowModal() == wx.ID_OK:
        filepath = dialog.GetPath()
    else:
        filepath = ''
    return filepath


def load_error_analysis_data(directory):
    # Find all error analysis files
    error_analysis_files = []
    c = 1
    while True:
        filename = directory + 'error_surface_' + str(c) + '.dat'
        c += 1
        if os.path.exists(filename):
            error_analysis_files.append(filename)
        else:
            break
    n_files = len(error_analysis_files)
    # Set the error analysis parameters
    error_analysis_parameters = []
    for i in range(n_files):
        file = open(error_analysis_files[i], 'r')
        head = str(file.readline())
        column_names = wrap(head, 20)
        n_parameters = len(column_names) - 1
        subset_error_analysis_parameters = []
        for j in range(n_parameters):
            parameter_name = column_names[j].split()
            name = parameter_name[0].strip()
            component = int(parameter_name[1]) - 1
            name_found = False 
            for item in const['model_parameter_names']:    
                if name == item:
                    name_found = True
            if not name_found:
                raise ValueError('Unknown parameter name was found!')
                sys.exit(1)
            parameter_id = ParameterID(name, component)
            subset_error_analysis_parameters.append(parameter_id)
            file.close()
        error_analysis_parameters.append(subset_error_analysis_parameters)    
    # Read the error analysis data
    error_surfaces = []
    for i in range(n_files):
        file = open(error_analysis_files[i], 'r')
        n_points = len(file.readlines()) - 1
        file.close()
        n_parameters = len(error_analysis_parameters[i])
        error_surface = {}
        error_surface['parameters'] = np.zeros([n_parameters,n_points])
        error_surface['chi2'] = np.zeros(n_points)
        file = open(error_analysis_files[i], 'r')
        next(file)
        c = 0
        for line in file:
            data = wrap(line, 20)
            for j in range(n_parameters):
                parameter_name = error_analysis_parameters[i][j].name
                parameter_value = float(data[j]) * const['model_parameter_scales'][parameter_name]
                error_surface['parameters'][j][c] = parameter_value
            error_surface['chi2'][c] = float(data[-1])
            c += 1
        error_surfaces.append(error_surface)
        file.close()
    return error_analysis_parameters, error_surfaces


def enter_model_parameters(fitting_parameters):
    c = 0
    optimized_model_parameters_all_runs = []
    while True:
        print('\nOptimized values of the fitting parameters: solution {0}'.format(c+1))
        optimized_model_parameters = []
        for parameter_name in const['model_parameter_names']:
            parameter_indices = fitting_parameters['indices'][parameter_name]
            for i in range(len(parameter_indices)):
                parameter_object = parameter_indices[i]
                if parameter_object.optimize:
                    var = input('Optimized value of ' + const['model_parameter_names_and_units'][parameter_name] + ', mode ' + str(i+1) + ': ')
                    val = str(var)
                    optimized_model_parameter = float(val) 
                    print('{0}'.format(optimized_model_parameter))
                optimized_model_parameters.append(optimized_model_parameter)
        optimized_model_parameters_all_runs.append(optimized_model_parameters)
        var = input("\nDo you want to enter another set of optimized fitting parameters (y or n)? ")
        val = str(var)
        if val == 'y':
            print('Answer: {0}'.format('yes'))
            c += 1
        else:
            print('Answer: {0}'.format('no'))
            break
    if len(optimized_model_parameters_all_runs) > 1:
        var = input("\nWhich set of optimized fitting parameters was used to record the error surfaces? ")
        val = int(var)
        idx_best_solution = val - 1
        print('{0}'.format(idx_best_solution+1))
    else:
        idx_best_solution = 0
    return optimized_model_parameters_all_runs, idx_best_solution
           

def load_model_parameters(directory, solutions_of_all_runs):
    filepath = directory + 'fitting_parameters.dat'
    optimized_model_parameters, model_parameter_errors, optimized_model_parameters_all_runs = load_optimized_model_parameters(filepath)
    if solutions_of_all_runs:
        filename = directory + 'logfile.log'
        file = open(filename, 'r')
        lines = file.readlines()
        for line in lines:
            if line.find('The best solution was found in optimization run no.') != -1:
                content = list(line.split())
                idx_best_solution = int(content[9]) - 1 
        file.close()    
    else:
        optimized_model_parameters_all_runs = [optimized_model_parameters]
        idx_best_solution = 0
    return optimized_model_parameters_all_runs, idx_best_solution


def enter_chi2_threshold():
    var = input("\nEnter the chi-squared threshold: ")
    chi2_threshold = float(var)
    print('{0}'.format(chi2_threshold))
    
    var = input("\nEnter the minimum chi-squared: ")
    chi2_minimum = float(var)
    print('{0}'.format(chi2_minimum))
    
    return chi2_threshold, chi2_minimum


def load_chi2_threshold(directory):

    filename = directory + 'logfile.log'
    file = open(filename, 'r')
    lines = file.readlines()
    for line in lines:
        if line.find('Total chi-squared threshold') != -1:
            content = list(line.split())
            chi2_threshold = float(content[4])      
    file.close()
    
    file = open(filename, 'r')
    lines = file.readlines()
    for line in lines:
        if line.find('Minimum chi-squared') != -1:
            content = list(line.split())
            chi2_minimum = float(content[2])      
    file.close()
    
    print('\nChi-squared threshold: {0}'.format(chi2_threshold))
    print('\nMinimum chi-squared: {0}'.format(chi2_minimum))
    return chi2_threshold, chi2_minimum


if __name__ == '__main__':
    
    # Load the content of the config file
    filepath_config = get_filepath("Open the configuration file...")
    with io.open(filepath_config) as file:
        config = libconf.load(file)
    fitting_parameters = read_fitting_parameters(config)
    error_analyzer = read_error_analysis_settings(config, mode={'fitting':1, 'error_analysis':1})
    print('\nConfiguration file was loaded from:\n{0}'.format(filepath_config))

    # Load the results of the error analysis
    var = input("\nAre the error analysis results located in the different directory than the configuration file (y or n)? ")
    val = str(var)
    if val == 'y':
        print('Answer: {0}'.format('yes'))
        filepath_error_analysis = get_filepath("Open the directory with the error analysis results...") 
    else:
        print('Answer: {0}'.format('no'))
        filepath_error_analysis = filepath_config
    directory_error_analysis = os.path.dirname(filepath_error_analysis) + '/'
    error_analysis_parameters, error_surfaces = load_error_analysis_data(directory_error_analysis)
    # Init logger, data saver, and data plotter
    data_saver = DataSaver(True, True)
    data_saver.create_output_directory(directory_error_analysis, 'error_analysis_results****')
    plotter = Plotter(data_saver)
    sys.stdout = Logger(data_saver.directory+'logfile.log')
    print('\nError surfaces were loaded from:\n{0}'.format(directory_error_analysis))

    # Load the optimized fitting parameters
    var = input("\nDo you want to enter the optimized values of the fitting parameters manually (y or n)? ")
    val = str(var)
    if val == 'y':
        print('Answer: {0}'.format('yes'))
        optimized_model_parameters_all_runs, idx_best_solution = enter_model_parameters(fitting_parameters)
    else:
        print('Answer: {0}'.format('no'))
        var = input("\nAre the fitting results located in the different directory than the error analysis results (y or n)? ")
        val = str(var)
        if val == 'y':
            print('Answer: {0}'.format('yes'))
            filepath_fitting = get_filepath("\nOpen the directory with the fitting results...")
            directory_fitting = os.path.dirname(filepath_fitting) + '/'
        else:
            print('Answer: {0}'.format('no'))
            directory_fitting = directory_error_analysis
        print('\nThe optimized model parameters were loaded from:\n{0}'.format(directory_fitting))
        
        var = input("\nDo you want to include the solutions of all fitting runs (y or n)? ")
        val = str(var)
        if val == 'y':
            print('Answer: {0}'.format('yes'))
            solutions_of_all_runs = True
        else:
            print('Answer: {0}'.format('no'))
            solutions_of_all_runs = False
        optimized_model_parameters_all_runs, idx_best_solution = load_model_parameters(directory_fitting, solutions_of_all_runs)
    
    # Load the chi2 threshold
    var = input("\nDo you want to enter the chi-squared threshold manually (y or n)? ")
    val = str(var)
    if val == 'y':
        print('Answer: {0}'.format('yes'))
        chi2_threshold, chi2_minimum = enter_chi2_threshold()
    else:
        print('Answer: {0}'.format('no'))
        chi2_threshold, chi2_minimum = load_chi2_threshold(directory_error_analysis)
    chi2_thresholds = chi2_threshold * np.ones(len(error_analysis_parameters))
    
    # Recompute the error profiles
    error_profiles = error_analyzer.compute_error_profiles(error_surfaces)
    optimized_model_parameters = optimized_model_parameters_all_runs[idx_best_solution]
    error_surfaces_2d = error_analyzer.compute_2d_error_surfaces(error_surfaces, True)
    error_profiles = error_analyzer.correct_error_profiles(error_profiles, chi2_minimum, optimized_model_parameters, error_analysis_parameters, fitting_parameters)
    
    # Recompute the parameter errors
    print("Recalculating the errors of the model parameters... ")
    error_analyzer.background_errors = False
    model_parameter_errors, model_parameter_uncertainty_interval_bounds = \
            error_analyzer.compute_model_parameter_errors(optimized_model_parameters_all_runs[idx_best_solution],
                                                          error_profiles,
                                                          chi2_minimum,
                                                          chi2_thresholds,
                                                          error_analysis_parameters,
                                                          fitting_parameters)
    print_model_parameters(optimized_model_parameters_all_runs[idx_best_solution], 
                           model_parameter_errors, 
                           fitting_parameters)
    data_saver.save_model_parameters(optimized_model_parameters_all_runs[idx_best_solution], 
                                     model_parameter_errors, 
                                     fitting_parameters)
    
    # Plot data
    plotter.plot_error_surfaces(error_surfaces, error_surfaces_2d, optimized_model_parameters_all_runs[idx_best_solution], error_analysis_parameters, fitting_parameters, chi2_minimum, chi2_thresholds)
    plotter.plot_error_surfaces_multiple_runs(error_surfaces, error_surfaces_2d, optimized_model_parameters_all_runs, error_analysis_parameters, fitting_parameters, chi2_minimum, chi2_thresholds)
    plotter.plot_error_profiles(error_profiles, optimized_model_parameters_all_runs[idx_best_solution], error_analysis_parameters, fitting_parameters, 
                                model_parameter_uncertainty_interval_bounds, chi2_minimum, chi2_thresholds) 
    