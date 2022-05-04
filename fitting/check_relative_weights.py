import sys
import copy
import numpy as np
from supplement.definitions import const


def check_relative_weights(parameters_indices, optimized_parameters_values, fixed_parameters_values):
    ''' Check that the sum of relative weights assigned to different components does not exceed 1 (for each spin pair)'''
    new_optimized_parameters_values = copy.deepcopy(optimized_parameters_values)
    parameter_indices = parameters_indices['rel_prob']
    for i in range(len(parameter_indices)):
        # Calculate the sum of 'rel_prob'
        sum_fitted = 0
        sum_fixed = 0
        for j in range(len(parameter_indices[i])):
            parameter_object = parameter_indices[i][j]
            if parameter_object.optimize:
                parameter_value = new_optimized_parameters_values[parameter_object.index]
                sum_fitted += parameter_value
            else:
                parameter_value = fixed_parameters_values[parameter_object.index]
                sum_fixed += parameter_value
        # Check that the sum of 'rel_prob' does not exceed 1. If it does, normalize all 'rel_prob' such that the sum is exactly 1.
        if (sum_fitted + sum_fixed) > 1:
            max_sum_fitted = 1 - sum_fixed
            for j in range(len(parameter_indices[i])):
                parameter_object = parameter_indices[i][j]
                if parameter_object.optimize:
                    new_optimized_parameters_values[parameter_object.index] *= max_sum_fitted / sum_fitted
    return new_optimized_parameters_values