import copy
import numpy as np


def check_relative_weights(optimized_parameters, fitting_parameters):
    ''' Checks out that the sum of relative weights assigned to different components does not exceed 1 (for each spin pair)'''
    optimized_parameters_array = np.array(optimized_parameters)
    if optimized_parameters_array.ndim == 1:
        new_optimized_parameters = copy.deepcopy(optimized_parameters)
        parameter_indices = fitting_parameters['indices']['rel_prob']
        # Calculate the sum of 'rel_prob'
        sum_optimized = 0
        sum_fixed = 0
        for i in range(len(parameter_indices)):
            parameter_object = parameter_indices[i]
            if parameter_object.optimize:
                parameter_value = new_optimized_parameters[parameter_object.index]
                sum_optimized += parameter_value
            else:
                parameter_value = fitting_parameters['values'][parameter_object.index]
                sum_fixed += parameter_value
        # Check that the sum of 'rel_prob' does not exceed 1. If it does, normalize all 'rel_prob' such that the sum is exactly 1.
        if (sum_optimized + sum_fixed) > 1:
            max_sum_optimized = 1 - sum_fixed
            for i in range(len(parameter_indices)):
                parameter_object = parameter_indices[i]
                if parameter_object.optimize:
                    new_optimized_parameters[parameter_object.index] *= max_sum_optimized / sum_optimized
        return new_optimized_parameters
    else:
        new_optimized_parameters = copy.deepcopy(optimized_parameters)
        for parameter_values in new_optimized_parameters:
            parameter_indices = fitting_parameters['indices']['rel_prob']
            # Calculate the sum of 'rel_prob'
            sum_optimized = 0
            sum_fixed = 0
            for i in range(len(parameter_indices)):
                parameter_object = parameter_indices[i]
                if parameter_object.optimize:
                    parameter_value = parameter_values[parameter_object.index]
                    sum_optimized += parameter_value
                else:
                    parameter_value = fitting_parameters['values'][parameter_object.index]
                    sum_fixed += parameter_value
            # Check that the sum of 'rel_prob' does not exceed 1. If it does, normalize all 'rel_prob' such that the sum is exactly 1.
            if (sum_optimized + sum_fixed) > 1:
                max_sum_optimized = 1 - sum_fixed
                for i in range(len(parameter_indices)):
                    parameter_object = parameter_indices[i]
                    if parameter_object.optimize:
                        parameter_values[parameter_object.index] *= max_sum_optimized / sum_optimized       
        return new_optimized_parameters