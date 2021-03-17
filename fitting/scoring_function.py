import sys
from mathematics.chi2 import chi2


def merge_fitted_and_fixed_variables(variables_indices, fitted_variables_values, fixed_variables_values):
    ''' Merge fitted and fixed variables into a single dictionary '''
    all_variables = {}
    for variable_name in variables_indices:
        variable_indices = variables_indices[variable_name]
        variable_list = []
        for i in range(len(variable_indices)):
            variable_sublist = []
            for j in range(len(variable_indices[i])):
                variable_id = variable_indices[i][j]
                if variable_id.opt:
                    variable_value = fitted_variables_values[variable_id.idx]
                else:
                    variable_value = fixed_variables_values[variable_id.idx]
                variable_sublist.append(variable_value)
            variable_list.append(variable_sublist)
        all_variables[variable_name] = variable_list
    return all_variables

	
def scoring_function(variables, **kwargs):
    ''' Scoring function '''
    # Read out the arguments
    simulator = kwargs.get('simulator', None)     
    experiments = kwargs.get('experiments', []) 
    spins = kwargs.get('spins', [])
    fitting_parameters = kwargs.get('fitting_parameters', {})
    if (simulator == None) or (experiments == []) or (spins == []) or (fitting_parameters == []):
        raise ValueError('Scoring function has got inappropriate arguments!')
        sys.exit(1)
    # Merge fitted variables and fixed variables into a single dictionary
    all_variables = merge_fitted_and_fixed_variables(fitting_parameters['indices'], variables, fitting_parameters['values'])
    # Simulate PDS time traces
    simulated_time_traces, modulation_depth_scale_factors = simulator.compute_time_traces(experiments, spins, all_variables, False)
    # Calculate the score
    total_score = 0.0
    if simulator.scale_chi2_by_modulation_depth:
        sum_modulation_depth_scale_factors = sum(modulation_depth_scale_factors)
    for i in range(len(experiments)):   
        score = chi2(simulated_time_traces[i]['s'], experiments[i].s, experiments[i].noise_std)
        if simulator.scale_chi2_by_modulation_depth:
            score = score * modulation_depth_scale_factors[i] / sum_modulation_depth_scale_factors
        total_score += score
    return total_score


def fit_function(variables, **kwargs):
    ''' Compute the fit to the experimental PDS time traces '''
    print('\nComputing the fit to the experimental time traces...') 
    # Read out the arguments
    simulator = kwargs.get('simulator', None)     
    experiments = kwargs.get('experiments', []) 
    spins = kwargs.get('spins', [])
    fitting_parameters = kwargs.get('fitting_parameters', {})
    if (simulator == None) or (experiments == []) or (spins == []) or (fitting_parameters == []):
        raise ValueError('Scoring function has got inappropriate arguments!')
        sys.exit(1)
    # Merge fitted variables and fixed variables into a single dictionary
    all_variables = merge_fitted_and_fixed_variables(fitting_parameters['indices'], variables, fitting_parameters['values'])
    # Simulate PDS time traces
    simulated_time_traces, modulation_depth_scale_factors = simulator.compute_time_traces(experiments, spins, all_variables, False)
    return simulated_time_traces, modulation_depth_scale_factors