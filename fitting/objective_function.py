import sys
from mathematics.chi2 import chi2


def merge_fitted_and_fixed_variables(variables_indices, fitted_variables_values, fixed_variables_values):
    ''' Merge fitted and fixed variables into a single dictionary '''
    all_variables = {}
    for variable_name in variables_indices:
        variable_indices = variables_indices[variable_name]
        list_variable_values = []
        for i in range(len(variable_indices)):
            sublist_variable_values = []
            for j in range(len(variable_indices[i])):
                variable_object = variable_indices[i][j]
                if variable_object.optimize:
                    variable_value = fitted_variables_values[variable_object.index]
                else:
                    variable_value = fixed_variables_values[variable_object.index]
                sublist_variable_values.append(variable_value)
            list_variable_values.append(sublist_variable_values)
        all_variables[variable_name] = list_variable_values
    return all_variables


def compute_degrees_of_freedom(experiments, variables, fit_modulation_depth):
    ''' Computes the number of degrees of freedom ''' 
    N = 0
    for experiment in experiments:
        N += experiment.s.size
    p = len(variables)
    if fit_modulation_depth:
        p += len(experiments)
    return (N-p)


def fit_function(variables, simulator, experiments, spins, fitting_parameters):
    ''' Compute the fit to the experimental PDS time traces '''
    # Merge fitted variables and fixed variables into a single dictionary
    all_variables = merge_fitted_and_fixed_variables(fitting_parameters['indices'], variables, fitting_parameters['values'])
    # Simulate PDS time traces
    simulated_time_traces, modulation_depth_scale_factors = simulator.compute_time_traces(experiments, spins, all_variables, False)
    return simulated_time_traces, modulation_depth_scale_factors


def objective_function(variables, simulator, experiments, spins, fitting_parameters, goodness_of_fit):
    ''' Objective function '''
    # Compute the fit
    simulated_time_traces, modulation_depth_scale_factors = fit_function(variables, simulator, experiments, spins, fitting_parameters)
    # Compute the score
    if goodness_of_fit == 'chi2':
        total_score = 0.0
        for i in range(len(experiments)):   
            total_score += chi2(simulated_time_traces[i]['s'], experiments[i].s, experiments[i].noise_std)
        return total_score
    elif goodness_of_fit == 'reduced_chi2':
        total_score = 0.0
        degrees_of_freedom = compute_degrees_of_freedom(experiments, variables, simulator.fit_modulation_depth)
        for i in range(len(experiments)):   
            total_score += chi2(simulated_time_traces[i]['s'], experiments[i].s, experiments[i].noise_std)
        return total_score / float(degrees_of_freedom) 
    elif goodness_of_fit == 'chi2_noise_std_1':
        total_score = 0.0
        for i in range(len(experiments)):   
            total_score += chi2(simulated_time_traces[i]['s'], experiments[i].s)
        return total_score
    elif goodness_of_fit == 'chi2_weighted_by_modulation_depth':
        total_score = 0.0
        for i in range(len(experiments)):   
            total_score += modulation_depth_scale_factors[i] * chi2(simulated_time_traces[i]['s'], experiments[i].s)
        return total_score 