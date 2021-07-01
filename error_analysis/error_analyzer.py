import sys
import time
import datetime
import numpy as np
import scipy
from multiprocessing import Pool
from input.load_optimized_parameters import load_optimized_parameters


class ErrorAnalyzer():
    ''' Error Analysis class '''

    def __init__(self, error_analysis_parameters):
        self.sample_size = error_analysis_parameters['sample_size']
        self.confidence_interval = error_analysis_parameters['confidence_interval']
        self.filepath_optimized_parameters = error_analysis_parameters['filepath_optimized_parameters']
        self.objective_function = None
    
    def set_objective_function(self, func):
        ''' Set an objective function '''
        self.objective_function = func
        
    def load_optimized_parameters(self, parameter_indices):
        ''' Loads the optimized values of fitting parameters from a file '''
        if self.filepath_optimized_parameters != '':
            return load_optimized_parameters(self.filepath_optimized_parameters)
        else:
            raise ValueError('No file with the optimized parameters was provided!')
            sys.exit(1)
    
    def run_error_analysis(self, error_analysis_parameters, fitting_parameters, optimized_parameters):
        ''' Runs error analysis '''
        if optimized_parameters != []:
            print('\nStarting the error analysis...')
            time_start = time.time()
            numerical_error = self.compute_numerical_error(optimized_parameters)
            score_threshold = self.compute_score_threshold(self.confidence_interval, numerical_error, 1)
            score_vs_parameter_subsets = self.compute_score_vs_parameter_subsets(error_analysis_parameters, fitting_parameters, optimized_parameters)
            score_vs_parameters = self.compute_score_vs_parameters(score_vs_parameter_subsets)
            parameter_errors = self.compute_parameters_errors(error_analysis_parameters, score_vs_parameter_subsets, score_threshold, fitting_parameters, optimized_parameters)
            time_finish = time.time()
            time_elapsed = str(datetime.timedelta(seconds = time_finish - time_start))
            print('The error analysis is finished. Total duration: {0}'.format(time_elapsed))
            return score_vs_parameter_subsets, score_vs_parameters, numerical_error, score_threshold, parameter_errors
   
    def compute_numerical_error(self, optimized_parameters):
        ''' Computes the numerical error '''
        print('Computing the numerical error...')
        # Make multiple copies of the optimized fitting parameters
        variables = np.tile(optimized_parameters, (self.sample_size, 1))
        # Calculate the score
        self.pool = Pool()
        score = self.pool.map(self.objective_function, variables)
        self.pool.close()
        self.pool.join()
        # Compute the variation of the score
        score_min = min(score)
        score_max = max(score)
        # Compute the numerical error
        numerical_error = score_max - score_min
        print('Numerical error (chi2) = {0:<10.3}'.format(numerical_error))
        return numerical_error

    def compute_score_threshold(self, confidence_interval, numerical_error, degree_of_freedom):
        ''' Computes the score threshold '''
        print('Computing the score threshold...')
        score_threshold = 0.0
        if degree_of_freedom == 1:
            score_threshold = confidence_interval**2 + numerical_error
        else:
            p = 1.0 - scipy.stats.chi2.sf(confidence_interval**2, 1)
            score_threshold = scipy.stats.chi2.ppf(p, int(degree_of_freedom)) + numerical_error
        print('Score threshold (chi2) = {0:<10.3}'.format(score_threshold))
        return score_threshold

    def compute_score_vs_parameter_subsets(self, error_analysis_parameters, fitting_parameters, optimized_parameters):
        ''' Computes the score as a function of a sub-set of fitting parameters '''
        print('Computing the score as a function of fitting parameters ...')
        score_vs_parameter_subsets = []
        num_parameter_sets = len(error_analysis_parameters)
        for i in range(num_parameter_sets):
            sys.stdout.write('\r')
            sys.stdout.write('Parameter set {0} / {1}'.format(i+1, num_parameter_sets))
            sys.stdout.flush()
            score_vs_parameter_subset = {}  
            score_vs_parameter_subset['parameters'] = []
            # Make multiple copies of the optimized fitting parameters
            variables = np.tile(optimized_parameters, (self.sample_size, 1))
            # Vary the values of error analysis parameters
            num_parameters = len(error_analysis_parameters[i])
            for j in range(num_parameters):
                parameter_id = error_analysis_parameters[i][j]
                parameter_index = parameter_id.get_index(fitting_parameters['indices'])
                parameter_range = fitting_parameters['ranges'][parameter_index]
                parameter_lower_bound = parameter_range[0]
                parameter_upper_bound = parameter_range[1]
                parameter_values = parameter_lower_bound + (parameter_upper_bound - parameter_lower_bound) * np.random.rand(self.sample_size)
                variables[:,parameter_index] = parameter_values.reshape((1, self.sample_size))
                score_vs_parameter_subset['parameters'].append(parameter_values)
            # Compute the score    
            self.pool = Pool()
            score = self.pool.map(self.objective_function, variables)
            self.pool.close()
            self.pool.join()
            score_vs_parameter_subset['score'] = np.array(score)
            
            # for k in range(self.sample_size):
                # sys.stdout.write('\n')
                # for j in range(num_parameters):
                    # sys.stdout.write('{:10.4f}'.format(score_vs_parameter_subset['parameters'][j][k]*180/np.pi))
                # sys.stdout.write('{:20.4f}'.format(score_vs_parameter_subset['score'][k]))
            # sys.stdout.write('\n')
            
            score_vs_parameter_subsets.append(score_vs_parameter_subset)
        sys.stdout.write('\n')
        return score_vs_parameter_subsets  
    
    def compute_score_vs_parameters(self, score_vs_parameter_subsets, num_points = 100):
        ''' Computes the score as a function of individual fitting parameters '''
        score_vs_parameters = []
        for i in range(len(score_vs_parameter_subsets)):
            parameter_subset = score_vs_parameter_subsets[i]['parameters']
            for j in range(len(parameter_subset)):
                parameter_values = parameter_subset[j]
                score_values = score_vs_parameter_subsets[i]['score']
                # Make a parameter grid
                parameter_min = np.amin(parameter_values)
                parameter_max = np.amax(parameter_values)
                parameter_step = (parameter_max - parameter_min) / float(num_points-1)
                parameter_grid = np.linspace(parameter_min, parameter_max, num_points)
                # Find a minimal value of the score at each grid point
                minimized_score_values = np.zeros(num_points)
                indices_nonempty_bins = []
                for k in range(num_points):
                    indices_grid_point = np.where(np.abs(parameter_values-parameter_grid[k]) <= 0.5*parameter_step)[0]
                    if indices_grid_point != []:
                        minimized_score_values[k] = np.amin(score_values[indices_grid_point])
                        indices_nonempty_bins.append(k)
                parameter_grid = parameter_grid[indices_nonempty_bins]
                minimized_score_values = minimized_score_values[indices_nonempty_bins]
                score_vs_parameter = {}
                score_vs_parameter['parameter'] = parameter_grid
                score_vs_parameter['score'] = minimized_score_values
                score_vs_parameters.append(score_vs_parameter)
        return score_vs_parameters
                

    def compute_parameters_errors(self, error_analysis_parameters, score_vs_parameter_subsets, score_threshold, fitting_parameters, optimized_parameters):
        ''' Computes the uncernainty intervals of the optimized fitting parameters '''
        print('Computing the uncernainty intervals of the optimized fitting parameters...')
        parameter_errors = np.empty(optimized_parameters.size)
        parameter_errors[:] = np.nan
        for i in range(len(error_analysis_parameters)):
            for j in range(len(error_analysis_parameters[i])):
                parameter_id = error_analysis_parameters[i][j]
                parameter_index = parameter_id.get_index(fitting_parameters['indices'])
                optimized_parameter_value = optimized_parameters[parameter_index]
                parameter_values = score_vs_parameter_subsets[i]['parameters'][j]
                score_values = score_vs_parameter_subsets[i]['score']
                parameter_error = self.compute_parameter_error(parameter_values, optimized_parameter_value, score_values, score_threshold)
                if np.isnan(parameter_errors[parameter_index]):
                    parameter_errors[parameter_index] = parameter_error
                else:
                    if parameter_error > parameter_errors[parameter_index]:
                        parameter_errors[parameter_index] = parameter_error
        return parameter_errors
    
    def compute_parameter_error(self, parameter_values, optimized_parameter_value, score_values, score_threshold):
        ''' Computes the uncernainty interval of an optimized fitting parameter '''
        # Determine the minimal and maximal values of the parameter
        parameter_min = np.amin(parameter_values)
        parameter_max = np.amax(parameter_values)
        parameter_half_range = 0.5 * (parameter_max - parameter_min)
        # Determine the minimal score value
        minimal_score = np.amin(score_values)
        # Determine the parameter values which lie under the score threshold 
        selected_parameter_indices = np.where(score_values-minimal_score <= score_threshold)[0]
        selected_parameter_values = parameter_values[selected_parameter_indices]
        # Determine the uncertainty ranges of the parameter
        uncertainty_interval_lower_bound = np.amin(selected_parameter_values)
        uncertainty_interval_upper_bound = np.amax(selected_parameter_values)
        parameter_error_low_end = np.abs(optimized_parameter_value - uncertainty_interval_lower_bound)
        parameter_error_high_end = np.abs(optimized_parameter_value - uncertainty_interval_upper_bound)
        parameter_error = np.nan
        if (parameter_error_low_end < parameter_half_range) and (parameter_error_high_end < parameter_half_range):
            if (parameter_error_low_end > parameter_error_high_end):
                parameter_error = parameter_error_low_end
            else:
                parameter_error = parameter_error_high_end    
        return parameter_error   