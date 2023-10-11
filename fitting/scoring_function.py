import sys
import numpy as np
import copy
from mathematics.chi2 import chi2
from supplement.definitions import const


goodness_of_fit_parameters = ["chi2"]


def scoring_function(
    variables, goodness_of_fit, fitting_parameters, simulator, experiments, spins, 
    more_output, reset_field_orientations, fixed_params_included
    ):
    """Scoring function."""
    # Compute the fit
    if more_output: 
        simulated_time_traces, simulated_data = fit_function(
            variables, fitting_parameters, simulator, experiments, spins, 
            more_output, reset_field_orientations, fixed_params_included
            )
    else:
        simulated_time_traces = fit_function(
            variables, fitting_parameters, simulator, experiments, spins, 
            more_output, reset_field_orientations, fixed_params_included
            )
    # Compute the score
    if goodness_of_fit == "chi2":
        score = 0
        for i in range(len(experiments)):
            score += chi2(simulated_time_traces[i], experiments[i].s, experiments[i].noise_std)
    if more_output:
        return score, simulated_data
    else:
        return score


def fit_function(
    variables, fitting_parameters, simulator, experiments, spins, 
    more_output, reset_field_orientations, fixed_params_included 
    ):
    """Compute the fit to the PDS time traces."""
    if fixed_params_included:
        # Normalize relative weights
        normalized_variables = normalize_weights(variables, fitting_parameters)
        # Merge the optimized and fixed model parameters
        model_parameters = merge_optimized_and_fixed_parameters(normalized_variables, fitting_parameters)  
    else:
        model_parameters = variables
    # Simulate the PDS time traces
    if more_output:
        simulated_time_traces, simulated_data = simulator.simulate_time_traces(
            model_parameters, experiments, spins, more_output, reset_field_orientations, display_messages = False 
            )
        return simulated_time_traces, simulated_data
    else:
        simulated_time_traces = simulator.simulate_time_traces(
            model_parameters, experiments, spins, more_output, reset_field_orientations, display_messages = False)
        return simulated_time_traces


def normalize_weights(variables, fitting_parameters):
    """Normalize the relative weights assigned to different components of multimodal distributions."""
    if isinstance(variables, np.ndarray):
        normalized_variables = copy.deepcopy(variables)
    else:
        normalized_variables = np.array(variables)
    if normalized_variables.ndim == 1:
        # Calculate the sum of "rel_prob"
        sum_opt, sum_fixed = 0.0, 0.0
        for fitting_parameter in fitting_parameters["rel_prob"]:
            if fitting_parameter.optimized:
                parameter_value = normalized_variables[fitting_parameter.get_index()]
                sum_opt += parameter_value
            else:
                parameter_value = fitting_parameter.get_value()
                sum_fixed += parameter_value
        # Check that the sum of "rel_prob" does not exceed 1. 
        # If it does, normalize all "rel_prob" such that the sum is exactly 1.
        if (sum_opt + sum_fixed) > 1:
            max_sum_opt = 1 - sum_fixed
            for fitting_parameter in fitting_parameters["rel_prob"]:
                if fitting_parameter.optimized:
                    normalized_variables[fitting_parameter.get_index()] *= max_sum_opt / sum_opt
    else:
        for variable_set in normalized_variables:
            # Calculate the sum of "rel_prob"
            sum_opt, sum_fixed = 0.0, 0.0
            for fitting_parameter in fitting_parameters["rel_prob"]:
                if fitting_parameter.optimized:
                    parameter_value = variable_set[fitting_parameter.get_index()]
                    sum_opt += parameter_value
                else:
                    parameter_value = fitting_parameter.get_value()
                    sum_fixed += parameter_value
            # Check that the sum of "rel_prob" does not exceed 1. 
            # If it does, normalize all "rel_prob" such that the sum is exactly 1.
            if (sum_opt + sum_fixed) > 1:
                max_sum_opt = 1 - sum_fixed
                for fitting_parameter in fitting_parameters["rel_prob"]:
                    if fitting_parameter.optimized:
                        variable_set[fitting_parameter.get_index()] *= max_sum_opt / sum_opt     
    return normalized_variables
    
    
def merge_optimized_and_fixed_parameters(variables, fitting_parameters):
    """Merge the optimized and fixed model parameters into a single dictionary."""
    model_parameters = {}
    for parameter_name in const["model_parameter_names"]:
        parameter_values = []
        for fitting_parameter in fitting_parameters[parameter_name]:
            if fitting_parameter.optimized:
                parameter_value = variables[fitting_parameter.get_index()]
            else:
                parameter_value = fitting_parameter.get_value()
            parameter_values.append(parameter_value)
        model_parameters[parameter_name] = parameter_values
    return model_parameters