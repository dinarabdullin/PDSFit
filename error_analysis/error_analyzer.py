import sys
import time
import datetime
import numpy as np
from multiprocessing import Pool


class ErrorAnalyzer():
    ''' Error Analysis class '''

    def __init__(self, error_analysis_parameters):
        self.sample_size = error_analysis_parameters['sample_size']
        self.confidence_interval = error_analysis_parameters['confidence_interval']
        self.filepath_optimized_parameters = error_analysis_parameters['filepath_optimized_parameters']
    
    def run_error_analysis(self, error_analysis_parameters, optimized_parameters, fitting_parameters, objective_function):
        ''' Run error analysis '''
        if optimized_parameters != []:
            print('\nStarting the error analysis...')
            time_start = time.time()
            # Compute the numerical error
            numerical_error = self.compute_numerical_error(optimized_parameters, objective_function)
            # Compute the score threshold
            score_threshold = self.compute_score_threshold(self.confidence_interval, numerical_error, 1)
            # Calculate the score as a function of parameters
            score_vs_parameter_sets = self.compute_score_vs_parameter_sets(error_analysis_parameters, optimized_parameters, fitting_parameters, objective_function)
            # Calculate the errors of fitting parameters
            parameters_errors = self.compute_parameters_errors(error_analysis_parameters, score_vs_parameter_sets, score_threshold, optimized_parameters, fitting_parameters)
            # Timings
            time_finish = time.time()
            time_elapsed = str(datetime.timedelta(seconds = time_finish - time_start))
            print('The error analysis is finished. Total duration: {0}'.format(time_elapsed))
            return score_vs_parameter_sets, numerical_error, score_threshold, parameters_errors
    
    def compute_score_vs_parameter_sets(self, error_analysis_parameters, optimized_parameters, fitting_parameters, objective_function):
        ''' Computes the score as a function of fitting parameters '''
        print('Computing the score as a function of fitting parameters ...')
        score_vs_parameter_sets = []
        num_parameter_sets = len(error_analysis_parameters)
        for i in range(num_parameter_sets):
            sys.stdout.write('\r')
            sys.stdout.write('Parameter set {0} / {1}'.format(i+1, num_parameter_sets))
            sys.stdout.flush()
            score_vs_parameter_set = {}  
            score_vs_parameter_set['parameters'] = []
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
                score_vs_parameter_set['parameters'].append(parameter_values)
            # Compute the score    
            self.pool = Pool()
            score = self.pool.map(objective_function, variables)
            self.pool.close()
            self.pool.join()
            score_vs_parameter_set['score'] = score
            score_vs_parameter_sets.append(score_vs_parameter_set)
        sys.stdout.write('\n')
        return score_vs_parameter_sets  
    
    def compute_numerical_error(self, optimized_parameters, objective_function):
        ''' Computes the numerical error '''
        print('Computing the numerical error...')
        # Make multiple copies of the optimized fitting parameters
        variables = np.tile(optimized_parameters, (self.sample_size, 1))
        # Calculate the score
        self.pool = Pool()
        score = self.pool.map(objective_function, variables)
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
    
    def compute_parameters_errors(self, error_analysis_parameters, score_vs_parameter_sets, score_threshold, optimized_parameters, fitting_parameters):
        ''' Computes the uncernainty intervals of the optimized fitting parameters '''
        print('Computing the uncernainty intervals of the optimized fitting parameters...')
        parameters_errors = np.empty(optimized_parameters.size)
        parameters_errors[:] = np.nan
        num_parameter_sets = len(error_analysis_parameters)
        for i in range(num_parameter_sets):
            num_parameters = len(error_analysis_parameters[i])
            for j in range(num_parameters):
                parameter_id = error_analysis_parameters[i][j]
                parameter_index = parameter_id.get_index(fitting_parameters['indices'])
                parameter_values = score_vs_parameter_sets[i]['parameters'][j]
                score_values = score_vs_parameter_sets[i]['score']
                parameter_error = self.compute_parameter_error(parameter_values, score_values, score_threshold)
                if np.isnan(parameters_errors[parameter_index]):
                    parameters_errors[parameter_index] = parameter_error
                else:
                    if parameter_error > parameters_errors[parameter_index]:
                        parameters_errors[parameter_index] = parameter_error
        return parameters_errors
    
    def compute_parameter_error(self, parameter_values, score_values, score_threshold):
        ''' Computes the uncernainty interval of an optimized fitting parameter '''
        # Determine the minimal and maximal values of the parameter
        minimal_parameter_value = np.amin(parameter_values)
        maximal_parameter_value = np.amax(parameter_values)
        parameter_half_range = 0.5 * (maximal_parameter_value - minimal_parameter_value)
        # Determine the minimal score value and the corresponding parameter value
        minimal_score = np.amin(score_values)
        index_minimal_score = np.argmin(score_values)
        optimal_parameter_value = parameter_values[index_minimal_score]
        # Determine the parameter values which lie under the score threshold 
        selected_parameter_indices = np.where(score_values-minimal_score <= score_threshold)[0]
        selected_parameter_values = parameter_values[selected_parameter_indices]
        # Determine the uncertainty ranges of the parameter
        uncertainty_interval_lower_bound = np.amin(np.array(selected_parameter_values))
        uncertainty_interval_upper_bound = np.amax(np.array(selected_parameter_values))
        parameter_error_low_end = np.abs(optimal_parameter_value - uncertainty_interval_lower_bound)
        parameter_error_high_end = np.abs(optimal_parameter_value - uncertainty_interval_upper_bound)
        parameter_error = np.nan
        if (parameter_error_low_end < parameter_half_range) and (parameter_error_high_end < parameter_half_range):
            if (parameter_error_low_end > parameter_error_high_end):
                parameter_error = parameter_error_low_end
            else:
                parameter_error = parameter_error_high_end    
        return parameter_error   