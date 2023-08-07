import sys
import time
import datetime
import numpy as np
import scipy
from scipy import interpolate
from copy import deepcopy
from multiprocessing import Pool
try:
    import mpi4py
    from mpi4py.futures import MPIPoolExecutor
except:
    pass
from mpi.mpi_status import get_mpi
from input.load_optimized_model_parameters import load_optimized_model_parameters
from fitting.check_relative_weights import check_relative_weights
from mathematics.find_nearest import find_nearest


class ErrorAnalyzer():
    ''' Error Analysis '''

    def __init__(self):
        self.objective_function = None
        self.objective_function_with_background_record = None
        self.parameter_names = {
            'samples_per_parameter':                'int', 
            'samples_numerical_error':              'int', 
            'confidence_interval':                  'int',
            'confidence_interval_numerical_error':  'int', 
            'background_errors':                    'int',
            'filepath_optimized_parameters':        'str', 
            }

    def set_intrinsic_parameters(self, error_analysis_parameters):
        self.samples_per_parameter = error_analysis_parameters['samples_per_parameter']
        self.samples_numerical_error = error_analysis_parameters['samples_numerical_error']
        self.confidence_interval = error_analysis_parameters['confidence_interval']
        self.confidence_interval_numerical_error = error_analysis_parameters['confidence_interval_numerical_error']
        self.background_errors = error_analysis_parameters['background_errors']
        self.filepath_optimized_parameters = error_analysis_parameters['filepath_optimized_parameters']
    
    def set_objective_function(self, func):
        ''' Sets the objective function '''
        self.objective_function = func
    
    def set_objective_function_with_background_record(self, func):
        ''' Set the objective function that records the background parameters '''
        self.objective_function_with_background_record = func

    def load_optimized_model_parameters(self):
        ''' Loads the optimized values of model parameters from a file '''
        if self.filepath_optimized_parameters != '':
            return load_optimized_model_parameters(self.filepath_optimized_parameters)
        else:
            raise ValueError('No file with the optimized model parameters was provided!')
            sys.exit(1)
    
    def run_error_analysis(self, error_analysis_parameters, optimized_model_parameters, optimized_background_parameters, 
                           simulated_time_traces, background_time_traces, background, fitting_parameters, modulation_depths):      
        ''' Runs the error analysis '''
        if optimized_model_parameters != []:
            sys.stdout.write('\n########################################################################\
                              \n#                            Error analysis                            #\
                              \n########################################################################\n')
            sys.stdout.flush()
            time_start = time.time()
            chi2_thresholds, chi2_minimum  = self.compute_chi2_threshold(error_analysis_parameters, optimized_model_parameters)
            error_surfaces, higher_dimensions = self.compute_error_surfaces(optimized_model_parameters, error_analysis_parameters, fitting_parameters)
            error_surfaces_2d = self.compute_2d_error_surfaces(error_surfaces, higher_dimensions)
            error_profiles = self.compute_error_profiles(error_surfaces)
            error_profiles = self.correct_error_profiles(error_profiles, chi2_minimum, optimized_model_parameters, error_analysis_parameters, fitting_parameters)
            model_parameter_errors, model_parameter_uncertainty_interval_bounds = \
                self.compute_model_parameter_errors(optimized_model_parameters, error_profiles, chi2_minimum, chi2_thresholds, error_analysis_parameters, fitting_parameters)
            background_parameter_errors = []
            error_bars_background_time_traces = []
            if self.background_errors:
                background_parameter_errors = self.compute_background_parameter_errors(optimized_background_parameters,
                                                                                       error_surfaces, 
                                                                                       chi2_minimum, 
                                                                                       chi2_thresholds,
                                                                                       error_analysis_parameters)                                                                                       
                error_bars_background_time_traces = self.compute_error_bars_for_background_time_traces(background_time_traces,
                                                                                                       error_surfaces, 
                                                                                                       error_analysis_parameters, 
                                                                                                       chi2_minimum, 
                                                                                                       chi2_thresholds, 
                                                                                                       background, 
                                                                                                       modulation_depths)
            time_elapsed = str(datetime.timedelta(seconds = time.time() - time_start))
            sys.stdout.write('\nThe error analysis is finished. Duration: {0}\n'.format(time_elapsed))
            sys.stdout.flush()
            return model_parameter_errors, background_parameter_errors, error_surfaces, error_surfaces_2d, error_profiles, \
                   model_parameter_uncertainty_interval_bounds, chi2_minimum, chi2_thresholds, error_bars_background_time_traces

    def compute_chi2_threshold(self, error_analysis_parameters, optimized_model_parameters):
        ''' Computes the chi2 threshold '''
        sys.stdout.write('\nComputing the chi-squared threshold... ')
        sys.stdout.flush()
        # Compute the chi2 thresholds based on the degrees of freedom and confidence interval
        num_parameters = len(optimized_model_parameters)
        degrees_of_freedom = np.arange(1,num_parameters+1,1)
        chi2_thresholds_theory = self.compute_chi2_threshold_for_confidence_interval(degrees_of_freedom, self.confidence_interval)
         # Estimate the contribution of the numerical error to the chi2 threshold
        chi2_minimum_mean, chi2_minimum_std = self.compute_numerical_error(optimized_model_parameters)
        chi2_minimum = chi2_minimum_mean
        chi2_threshold_numerical_error = float(self.confidence_interval_numerical_error) * chi2_minimum_std
        # Compute the sum of the chi2 thresholds and the numerical error
        dimensions = []
        selected_chi2_thresholds_theory = []
        chi2_thresholds = []
        for i in range(len(error_analysis_parameters)):
            dimension = len(error_analysis_parameters[i])
            dimensions.append(dimension)
            total_chi2_threshold = chi2_threshold_numerical_error + chi2_thresholds_theory[num_parameters-dimension] 
            selected_chi2_thresholds_theory.append(chi2_thresholds_theory[num_parameters-dimension])
            chi2_thresholds.append(total_chi2_threshold)
        unequal_dimensions = list(set(dimensions))
        unequal_chi2_thresholds_theory = list(set(selected_chi2_thresholds_theory))
        unequal_chi2_thresholds = list(set(chi2_thresholds))
        sorted_dimensions = sorted(unequal_dimensions)
        sorted_chi2_thresholds_theory = [x for _, x in sorted(zip(unequal_dimensions, unequal_chi2_thresholds_theory))]
        sorted_chi2_thresholds = [x for _, x in sorted(zip(unequal_dimensions, unequal_chi2_thresholds))]
        # Display the results
        sys.stdout.write('done!\n')
        sys.stdout.write('Minimum chi-squared:                         {0:<.1f}\n'.format(chi2_minimum))
        sys.stdout.write('Numerical error contribution ({0:d}-sigma):      {1:<.1f}\n'.format(self.confidence_interval, chi2_threshold_numerical_error))
        sys.stdout.write('Theoretical chi-squared threshold ({0:d}-sigma): '.format(self.confidence_interval))
        for i in range(len(sorted_dimensions)):
            sys.stdout.write('{0:<.1f} ({1:d}d)'.format(sorted_chi2_thresholds_theory[i], sorted_dimensions[i]))
            if i < len(sorted_dimensions) - 1:
                sys.stdout.write(', ')
            else:
                sys.stdout.write('\n')
        sys.stdout.write('Total chi-squared threshold ({0:d}-sigma):       '.format(self.confidence_interval))
        for i in range(len(sorted_dimensions)):
            sys.stdout.write('{0:<.1f} ({1:d}d)'.format(sorted_chi2_thresholds[i], sorted_dimensions[i]))
            if i < len(sorted_dimensions) - 1:
                sys.stdout.write(', ')
            else:
                sys.stdout.write('\n')
        return chi2_thresholds, chi2_minimum

    def compute_chi2_threshold_for_confidence_interval(self, degrees_of_freedom, confidence_interval):
        ''' Computes the chi2 threshold based on the degrees of freedom and confidence interval '''
        chi2_thresholds = []
        for v in degrees_of_freedom:
            chi2_threshold = 0.0
            if v == 1:
                chi2_threshold = float(confidence_interval)**2
            else:
                p = 1.0 - scipy.stats.chi2.sf(float(confidence_interval)**2, 1)
                chi2_threshold = scipy.stats.chi2.ppf(p, int(v))
            chi2_thresholds.append(chi2_threshold)
        chi2_thresholds = np.array(chi2_thresholds)
        return chi2_thresholds
    
    def compute_numerical_error(self, optimized_model_parameters):
        ''' Computes the numerical error '''
        # Make multiple copies of the optimized model parameters
        model_parameters = []
        for i in range(self.samples_numerical_error):
            model_parameters.append(optimized_model_parameters)
        # Calculate chi2
        run_with_mpi = get_mpi()
        if run_with_mpi:
            with MPIPoolExecutor() as executor:
                result = executor.map(self.objective_function, model_parameters)
            chi2_values = list(result)
        else:
            pool = Pool()
            chi2_values = pool.map(self.objective_function, model_parameters)
            pool.close()
            pool.join()
        chi2_values = np.array(chi2_values)
        # Set the minimum chi2 and the chi2 threshold due to the numerical error
        chi2_minimum_mean = np.mean(chi2_values)
        chi2_minimum_std = np.std(chi2_values)
        # # Plot
        # from plots.error_analysis.plot_numerical_error import plot_numerical_error
        # plot_numerical_error(chi2_values, chi2_minimum_mean, chi2_minimum_std)
        return chi2_minimum_mean, chi2_minimum_std

    def compute_error_surfaces(self, optimized_model_parameters, error_analysis_parameters, fitting_parameters):
        ''' Computes error surfaces '''
        sys.stdout.write('\nComputing error surfaces for the fitting parameters...\n')
        sys.stdout.flush()
        error_surfaces = []
        higher_dimensions = False
        num_parameter_sets = len(error_analysis_parameters)
        for i in range(num_parameter_sets):
            sys.stdout.write('\r')
            sys.stdout.write('Parameter set {0} / {1}'.format(i+1, num_parameter_sets))
            sys.stdout.flush()
            # Set the values of error analysis parameters
            num_parameters = len(error_analysis_parameters[i])
            if num_parameters == 1:
                num_samples = self.samples_per_parameter
                # Make multiple copies of the optimized model parameters
                model_parameters = np.tile(optimized_model_parameters, (num_samples, 1))
                # Vary the error analysis parameters
                parameter_id = error_analysis_parameters[i][0]
                parameter_index = parameter_id.get_index(fitting_parameters['indices'])
                parameter_range = fitting_parameters['ranges'][parameter_index]
                parameter_lower_bound = parameter_range[0]
                parameter_upper_bound = parameter_range[1]
                parameter_values = np.linspace(parameter_lower_bound, parameter_upper_bound, num=num_samples)
                model_parameters[:,parameter_index] = parameter_values.reshape((1, num_samples))
            else:
                if num_parameters > 2:
                    higher_dimensions = True
                num_samples = np.power(self.samples_per_parameter, num_parameters)
                # Make multiple copies of the optimized model parameters
                model_parameters = np.tile(optimized_model_parameters, (num_samples, 1)) 
                # Vary the error analysis parameters
                parameter_set = []
                for j in range(num_parameters):
                    parameter_id = error_analysis_parameters[i][j]
                    parameter_index = parameter_id.get_index(fitting_parameters['indices'])
                    parameter_range = fitting_parameters['ranges'][parameter_index]
                    parameter_lower_bound = parameter_range[0]
                    parameter_upper_bound = parameter_range[1]
                    parameter_values = np.linspace(parameter_lower_bound, parameter_upper_bound, num=self.samples_per_parameter)
                    parameter_set.append(parameter_values)
                parameter_grid = np.array(np.meshgrid(*parameter_set))
                parameter_grid_points = parameter_grid.reshape(num_parameters,-1).T
                for k in range(num_samples):
                    for j in range(num_parameters):
                        parameter_id = error_analysis_parameters[i][j]
                        parameter_index = parameter_id.get_index(fitting_parameters['indices'])
                        model_parameters[k,parameter_index] = parameter_grid_points[k][j]
            # Check / correct the relative weights
            for k in range(self.samples_per_parameter):
                model_parameters[k,:] = check_relative_weights(model_parameters[k,:], fitting_parameters)
            # Compute chi2 values
            run_with_mpi = get_mpi()
            if run_with_mpi:
                if self.background_errors:
                    with MPIPoolExecutor() as executor:
                        result = zip(*executor.map(self.objective_function_with_background_record, model_parameters))
                    chi2_values, background_parameters, modulation_depths = list(result)
                else:
                    with MPIPoolExecutor() as executor:
                        result = executor.map(self.objective_function, model_parameters)
                    chi2_values = list(result)
            else:
                pool = Pool()
                if self.background_errors:
                    chi2_values, background_parameters, modulation_depths = list(zip(*pool.map(self.objective_function_with_background_record, model_parameters)))
                else:
                    chi2_values = pool.map(self.objective_function, model_parameters)
                pool.close()
                pool.join()
            # Store the results
            error_surface = {}  
            error_surface['parameters'] = []
            for j in range(num_parameters):
                parameter_id = error_analysis_parameters[i][j]
                parameter_index = parameter_id.get_index(fitting_parameters['indices'])
                error_surface['parameters'].append(model_parameters[:,parameter_index])
            error_surface['chi2'] = np.array(chi2_values)
            if self.background_errors:
                error_surface['background_parameters'] = np.array(background_parameters)
                error_surface['modulation_depths'] = np.array(modulation_depths)
            error_surfaces.append(error_surface)
        sys.stdout.write('\ndone!\n')
        sys.stdout.flush()
        return error_surfaces, higher_dimensions 
    
    def compute_2d_error_surfaces(self, error_surfaces, higher_dimensions):
        ''' Computes 2d error surfaces '''
        error_surfaces_2d = []
        if higher_dimensions:
            sys.stdout.write('\nComputing 2d error surfaces from multi-dimensional error surfaces... ')
            sys.stdout.flush()
            for i in range(len(error_surfaces)):
                parameters = error_surfaces[i]['parameters']
                num_parameters = len(parameters)
                joint_error_surfaces = []
                if num_parameters > 2:
                    for k in range(num_parameters - 1):
                        for l in range(k+1, num_parameters):
                            parameter1_values = parameters[k]
                            parameter2_values = parameters[l]
                            chi2_values = error_surfaces[i]['chi2']
                            # Make parameter grids
                            n_points = self.samples_per_parameter
                            parameter1_min = np.amin(parameter1_values)
                            parameter1_max = np.amax(parameter1_values)
                            parameter1_step = (parameter1_max - parameter1_min) / (float(n_points)-1)
                            parameter1_grid = np.linspace(parameter1_min, parameter1_max, n_points)
                            parameter2_min = np.amin(parameter2_values)
                            parameter2_max = np.amax(parameter2_values)
                            parameter2_step = (parameter2_max - parameter2_min) / (float(n_points)-1)
                            parameter2_grid = np.linspace(parameter2_min, parameter2_max, n_points)
                            parameter1_2d_grid = np.array([[item] * n_points for item in parameter1_grid]).reshape(-1)
                            parameter2_2d_grid = np.array(parameter2_grid.tolist() * n_points)
                            # Find the minimum chi2 value at each grid point
                            minimized_chi2_values = np.zeros(n_points*n_points)
                            indices_nonempty_bins = []
                            for j in range(n_points*n_points):
                                indices_grid_points = np.where((np.abs(parameter1_values-parameter1_2d_grid[j]) <= 0.5*parameter1_step) & \
                                                               (np.abs(parameter2_values-parameter2_2d_grid[j]) <= 0.5*parameter2_step))[0] 
                                if len(indices_grid_points) > 0:
                                    minimized_chi2_values[j] = np.amin(chi2_values[indices_grid_points])
                                    indices_nonempty_bins.append(j)
                            parameter1_2d_grid = parameter1_2d_grid[indices_nonempty_bins]
                            parameter2_2d_grid = parameter2_2d_grid[indices_nonempty_bins]
                            minimized_chi2_values = minimized_chi2_values[indices_nonempty_bins]
                            # Store data
                            error_surface_2d = {}
                            error_surface_2d['parameters'] = [parameter1_2d_grid, parameter2_2d_grid]
                            error_surface_2d['chi2'] = minimized_chi2_values
                            joint_error_surfaces.append(error_surface_2d)
                error_surfaces_2d.append(joint_error_surfaces)
            sys.stdout.write('done!\n')
            sys.stdout.flush()
        return error_surfaces_2d 
    
    def compute_error_profiles(self, error_surfaces):
        ''' Computes error curves '''
        sys.stdout.write('\nComputing 1d error profiles from multi-dimensional error surfaces... ')
        sys.stdout.flush()
        error_profiles = []
        for i in range(len(error_surfaces)):
            parameters = error_surfaces[i]['parameters']
            for j in range(len(parameters)):
                parameter_values = parameters[j]
                chi2_values = error_surfaces[i]['chi2']
                # Make a parameter grid
                parameter_min = np.amin(parameter_values)
                parameter_max = np.amax(parameter_values)
                n_points = self.samples_per_parameter
                parameter_step = (parameter_max - parameter_min) / (float(n_points)-1)
                parameter_grid = np.linspace(parameter_min, parameter_max, n_points)
                # Find the minimum chi2 value at each grid point
                minimized_chi2_values = np.zeros(n_points)
                indices_nonempty_bins = []
                for k in range(n_points):
                    indices_grid_points = np.where(np.abs(parameter_values-parameter_grid[k]) <= 0.5*parameter_step)[0]
                    if len(indices_grid_points) > 0:
                        minimized_chi2_values[k] = np.amin(chi2_values[indices_grid_points])
                        indices_nonempty_bins.append(k)
                parameter_grid = parameter_grid[indices_nonempty_bins]
                minimized_chi2_values = minimized_chi2_values[indices_nonempty_bins]
                # Store data
                error_profile = {}
                error_profile['parameter'] = parameter_grid
                error_profile['chi2'] = minimized_chi2_values
                error_profiles.append(error_profile)
        sys.stdout.write('done!\n')
        sys.stdout.flush()
        return error_profiles  
    
    def correct_error_profiles(self, error_profiles, chi2_minimum, optimized_model_parameters, error_analysis_parameters, fitting_parameters):
        ''' '''
        count = 0
        for i in range(len(error_analysis_parameters)):
            parameter_uncertainty_interval_bounds_per_error_surface = []
            for j in range(len(error_analysis_parameters[i])):
                # Find the chi-squared minimum
                parameter_id = error_analysis_parameters[i][j]
                parameter_index = parameter_id.get_index(fitting_parameters['indices'])
                parameter_values = error_profiles[count]['parameter']
                chi2_values = error_profiles[count]['chi2']
                optimized_parameter_value = optimized_model_parameters[parameter_index]
                idx_optimized_parameter = find_nearest(parameter_values, optimized_parameter_value)
                if idx_optimized_parameter <= 1:
                    current_chi2_minimum = np.mean(chi2_values[0:5])
                elif idx_optimized_parameter == self.samples_per_parameter - 2:
                    current_chi2_minimum = np.mean(chi2_values[-5:])
                else:
                    current_chi2_minimum = np.mean(chi2_values[idx_optimized_parameter-2:idx_optimized_parameter+2])
                if current_chi2_minimum < chi2_minimum:
                    error_profiles[count]['chi2'] = error_profiles[count]['chi2'] + (chi2_minimum - current_chi2_minimum)
                count += 1  
        return error_profiles
    
    def compute_model_parameter_errors(self, optimized_model_parameters, error_profiles, chi2_minimum, chi2_thresholds, 
                                       error_analysis_parameters, fitting_parameters):
        ''' Computes the errors of the optimized model parameters '''
        sys.stdout.write('\nComputing the errors of the model parameters...\n')
        sys.stdout.flush()
        # Prepare the container for the model parameter errors
        model_parameter_errors = np.empty((optimized_model_parameters.size, 2,))
        model_parameter_errors[:] = np.nan
        model_parameter_uncertainty_interval_bounds = []
        # Compute the errors
        count = 0
        for i in range(len(error_analysis_parameters)):
            parameter_uncertainty_interval_bounds_per_error_surface = []
            for j in range(len(error_analysis_parameters[i])):
                # Find the chi2 values below the threshold
                parameter_id = error_analysis_parameters[i][j]
                parameter_index = parameter_id.get_index(fitting_parameters['indices'])
                optimized_parameter_value = optimized_model_parameters[parameter_index]
                parameter_values = error_profiles[count]['parameter']
                chi2_values = error_profiles[count]['chi2']
                parameter_min, parameter_max = np.amin(parameter_values), np.amax(parameter_values) 
                if len(parameter_values) < 100:
                    parameter_grid = np.linspace(parameter_min, parameter_max, 100)
                    parameter_step = (parameter_max - parameter_min) / 99
                    interpolation = interpolate.interp1d(parameter_values, chi2_values, kind='cubic')
                    chi2_grid = interpolation(parameter_grid)
                else:
                    parameter_grid = parameter_values
                    parameter_step = (parameter_max - parameter_min) / (float(self.samples_per_parameter) - 1)
                    chi2_grid = chi2_values
                selected_indices = np.where(chi2_grid <= chi2_minimum + chi2_thresholds[i])[0]
                if selected_indices.size == 0:
                    parameter_uncertainty_interval_bounds_per_error_surface.append([])
                else:
                    selected_parameter_values = parameter_grid[selected_indices]
                    parameter_uncertainty_interval, parameter_uncertainty_interval_bounds = \
                        self.compute_model_parameter_uncertainty_interval(parameter_id, optimized_parameter_value, selected_parameter_values, parameter_step)  
                    parameter_uncertainty_interval_bounds_per_error_surface.append(parameter_uncertainty_interval_bounds)
                    parameter_error = self.compute_model_parameter_error(optimized_parameter_value, parameter_uncertainty_interval, parameter_min, parameter_max, parameter_step)                
                    # Check whether the error was not calculated earlier and, if was, select the largest value
                    if np.isnan(model_parameter_errors[parameter_index][0]) and np.isnan(model_parameter_errors[parameter_index][1]):
                        model_parameter_errors[parameter_index][0], model_parameter_errors[parameter_index][1] = parameter_error[0], parameter_error[1]
                    else:
                        if parameter_error[0] < model_parameter_errors[parameter_index][0]:
                            model_parameter_errors[parameter_index][0] = parameter_error[0]
                        if parameter_error[1] > model_parameter_errors[parameter_index][1]:
                            model_parameter_errors[parameter_index][1] = parameter_error[1]
                count += 1
                # # Find the chi2 values below the threshold
                # chi2_values = error_profiles[count]['chi2']
                # selected_indices = np.where(chi2_values <= chi2_minimum + chi2_thresholds[i])[0]
                # if selected_indices.size == 0:
                    # parameter_uncertainty_interval_bounds_per_error_surface.append([])
                # else:
                    # parameter_id = error_analysis_parameters[i][j]
                    # parameter_index = parameter_id.get_index(fitting_parameters['indices'])
                    # optimized_parameter_value = optimized_model_parameters[parameter_index]
                    # parameter_values = error_profiles[count]['parameter']
                    # selected_parameter_values = parameter_values[selected_indices]
                    # parameter_min, parameter_max = np.amin(parameter_values), np.amax(parameter_values) 
                    # parameter_step = (parameter_max - parameter_min) / (float(self.samples_per_parameter) - 1)
                    # parameter_uncertainty_interval, parameter_uncertainty_interval_bounds = \
                        # self.compute_model_parameter_uncertainty_interval(parameter_id, optimized_parameter_value, selected_parameter_values, parameter_step)  
                    # parameter_uncertainty_interval_bounds_per_error_surface.append(parameter_uncertainty_interval_bounds)
                    # parameter_error = self.compute_model_parameter_error(optimized_parameter_value, parameter_uncertainty_interval, parameter_min, parameter_max, parameter_step)                
                    # # Check whether the error was not calculated earlier and, if was, select the largest value
                    # if np.isnan(model_parameter_errors[parameter_index][0]) and np.isnan(model_parameter_errors[parameter_index][1]):
                        # model_parameter_errors[parameter_index][0], model_parameter_errors[parameter_index][1] = parameter_error[0], parameter_error[1]
                    # else:
                        # if parameter_error[0] < model_parameter_errors[parameter_index][0]:
                            # model_parameter_errors[parameter_index][0] = parameter_error[0]
                        # if parameter_error[1] > model_parameter_errors[parameter_index][1]:
                            # model_parameter_errors[parameter_index][1] = parameter_error[1]
                # count += 1
            model_parameter_uncertainty_interval_bounds.append(parameter_uncertainty_interval_bounds_per_error_surface)
        sys.stdout.write('done!\n')
        sys.stdout.flush()
        return model_parameter_errors, model_parameter_uncertainty_interval_bounds         
    
    def compute_model_parameter_uncertainty_interval(self, parameter_id, optimized_parameter_value, parameter_values, parameter_step, delta=4):
        ''' Computes the uncertainty interval of a model parameter '''
        uncertainty_interval_bounds = []
        if parameter_values.size <= 1:
            sys.stdout.write('Warning: The uncertanty interval of parameter \'{0}\' is below the resolution of the error surface!\n'.format(parameter_id.name))
            sys.stdout.write('Reduce the parameter range or increase the number of points in the error surface.\n')
            sys.stdout.flush()
            uncertainty_interval = np.array([np.nan, np.nan])
        else: 
            # Check whether there are several uncertainty regions separated from each other
            parameter_scaled_values = parameter_values / parameter_step
            uncertainty_interval_lower_bounds, uncertainty_interval_upper_bounds = [], []
            uncertainty_interval_lower_bounds.append(parameter_scaled_values[0])
            for i in range(1, parameter_scaled_values.size - 1):
                if (parameter_scaled_values[i] - parameter_scaled_values[i-1] > delta):
                    uncertainty_interval_lower_bounds.append(parameter_scaled_values[i])
                    uncertainty_interval_upper_bounds.append(parameter_scaled_values[i-1]) 
            uncertainty_interval_upper_bounds.append(parameter_scaled_values[-1])        
            all_uncertainty_intervals = np.column_stack((uncertainty_interval_lower_bounds, uncertainty_interval_upper_bounds))   
            all_uncertainty_intervals *= parameter_step
            # Find the uncertainty interval that contains the optimized value of the model parameter
            uncertainty_interval = None
            for item in all_uncertainty_intervals:
                if (item[0] - parameter_step <= optimized_parameter_value) and \
                   (item[1] + parameter_step >= optimized_parameter_value):
                    uncertainty_interval = item
            if uncertainty_interval is None:
                sys.stdout.write('Warning: The optimized value of parameter \'{0}\' is outside the uncertanty interval!\n'.format(parameter_id.name))
                sys.stdout.flush()
                uncertainty_interval = np.array([np.nan, np.nan])
            else:
                uncertainty_interval_bounds.append(uncertainty_interval[0])
                uncertainty_interval_bounds.append(uncertainty_interval[1])  
                if uncertainty_interval[1] - uncertainty_interval[0] < parameter_step:
                    sys.stdout.write('Warning: The uncertanty interval of parameter \'{0}\' is below the resolution of the error surface!\n'.format(parameter_id.name))
                    sys.stdout.write('Reduce the parameter range or increase the number of points in the error surface.\n')
                    sys.stdout.flush()
                    uncertainty_interval = np.array([np.nan, np.nan])                                           
        uncertainty_interval_bounds = np.array(uncertainty_interval_bounds)
        return uncertainty_interval, uncertainty_interval_bounds
        
    def compute_model_parameter_error(self, optimized_parameter_value, uncertainty_interval, 
                                      parameter_min, parameter_max, parameter_step):
        ''' Computes the error of a model parameter based on its uncertainty interval '''
        if np.isnan(uncertainty_interval[0]) or np.isnan(uncertainty_interval[1]):
            parameter_minus_error, parameter_plus_error = np.nan, np.nan   
        else:
            if (uncertainty_interval[0] == parameter_min) and (uncertainty_interval[1] == parameter_max):
                parameter_minus_error, parameter_plus_error = np.nan, np.nan          
            else:
                parameter_minus_error = uncertainty_interval[0] - optimized_parameter_value
                parameter_plus_error = uncertainty_interval[1] - optimized_parameter_value
        return np.array([parameter_minus_error, parameter_plus_error]) 

    def compute_background_parameter_errors(self, optimized_background_parameters, error_surfaces, chi2_minimum, chi2_thresholds, error_analysis_parameters):
        ''' Computes the errors of the optimized background parameters '''
        sys.stdout.write('\nComputing the errors of the background parameters...\n')
        sys.stdout.flush()
        # Prepare the container for the background parameter errors
        background_parameter_errors = []
        for k in range(len(optimized_background_parameters)):
            background_parameter_errors_single_experiment = {}
            for key in optimized_background_parameters[k]:
                background_parameter_errors_single_experiment[key] = np.array([np.nan, np.nan])
            background_parameter_errors.append(background_parameter_errors_single_experiment)
        # Compute the errors
        for i in range(len(error_analysis_parameters)):
            # Find the chi2 values below the threshold
            chi2_values = error_surfaces[i]['chi2']
            selected_indices = np.where(chi2_values <= chi2_minimum + chi2_thresholds[i])[0]
            # Compute the errors of the background parameters
            if selected_indices.size != 0:
                for k in range(len(optimized_background_parameters)):
                    for key in optimized_background_parameters[k]:
                        # Optimized value of the background parameter
                        optimized_background_parameter_value = optimized_background_parameters[k][key]
                        # Background parameter values that correspond to the chi2 values below the chi2 threshold
                        background_parameter_values = []
                        n_samples = len(error_surfaces[i]['background_parameters'])
                        for l in range(n_samples):
                            background_parameter_values.append(error_surfaces[i]['background_parameters'][l][k][key])
                        background_parameter_values = np.array(background_parameter_values)
                        selected_background_parameter_values = background_parameter_values[selected_indices]
                        # Calculate the errors of the background parameter
                        background_parameter_uncertainty_interval_lower_bound = np.amin(selected_background_parameter_values)
                        background_parameter_uncertainty_interval_upper_bound = np.amax(selected_background_parameter_values)
                        background_parameter_minus_error = background_parameter_uncertainty_interval_lower_bound - optimized_background_parameter_value
                        background_parameter_plus_error = background_parameter_uncertainty_interval_upper_bound - optimized_background_parameter_value
                        background_parameter_error = np.array([background_parameter_minus_error, background_parameter_plus_error])
                        # Check whether the error was not calculated earlier and, if was, select the largest value
                        if np.isnan(background_parameter_errors[k][key][0]) and np.isnan(background_parameter_errors[k][key][1]):
                            background_parameter_errors[k][key][0], background_parameter_errors[k][key][1] = background_parameter_error[0], background_parameter_error[1]
                        else:
                            if background_parameter_error[0] < background_parameter_errors[k][key][0]:
                                background_parameter_errors[k][key][0] = background_parameter_error[0] 
                            if background_parameter_error[1] > background_parameter_errors[k][key][1]:
                                background_parameter_errors[k][key][1] = background_parameter_error[1]         
        sys.stdout.write('done!\n')
        sys.stdout.flush()
        return background_parameter_errors

    def compute_error_bars_for_background_time_traces(self, background_time_traces, error_surfaces, error_analysis_parameters, 
                                                      chi2_minimum, chi2_thresholds, background, modulation_depths):
        ''' Compute the error bars for the simulated backgrounds based on the error surfaces '''
        sys.stdout.write('\nComputing the error bars for the simulated PDS backgrounds... ')
        sys.stdout.flush()
        # Prepare the container for the error bars
        error_bars_background_time_traces = []
        for k in range(len(background_time_traces)):
            error_bars_background_time_trace = []
            for m in range(background_time_traces[k]['t'].size):
                error_bars_background_time_trace.append([0.0, 0.0])
            error_bars_background_time_traces.append(np.array(error_bars_background_time_trace))
        error_bars_background_time_traces = np.array(error_bars_background_time_traces)
        # Compute the error bars
        for i in range(len(error_analysis_parameters)):
            chi2_values = error_surfaces[i]['chi2']
            selected_indices = np.where(chi2_values <= chi2_minimum + chi2_thresholds[i])[0]
            background_parameter_test_sets = error_surfaces[i]['background_parameters'][selected_indices] 
            modulation_depths = error_surfaces[i]['modulation_depths'][selected_indices] 
            for j in range(len(background_parameter_test_sets)):
                for k in range(len(background_time_traces)):
                    background_time_trace = background_time_traces[k]
                    background_parameter_test_set = background_parameter_test_sets[j][k]
                    modulation_depth = modulation_depths[j][k]
                    test_time_trace = background.get_background(background_time_trace['t'], background_parameter_test_set, modulation_depth)
                    residuals = test_time_trace - background_time_trace['s']
                    for m in range(background_time_trace['t'].size):
                        if residuals[m] < error_bars_background_time_traces[k][m][0]:
                            error_bars_background_time_traces[k][m][0] = residuals[m]
                        if residuals[m] > error_bars_background_time_traces[k][m][1]:
                            error_bars_background_time_traces[k][m][1] = residuals[m]
        sys.stdout.write('done!\n')
        sys.stdout.flush()
        return error_bars_background_time_traces