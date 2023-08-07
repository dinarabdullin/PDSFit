import sys
from mathematics.chi2 import chi2
from fitting.merge_parameters import merge_parameters
from fitting.check_relative_weights import check_relative_weights


def fit_function(variables, simulator, experiments, spins, fitting_parameters, reset_field_orientations, fixed_variables_included, more_output):
    ''' Computes the fit to the PDS time traces '''
    if fixed_variables_included:
        # Check / correct the relative weights
        final_variables = check_relative_weights(variables, fitting_parameters)
        # Merge the optimized and fixed model parameters into a single dictionary
        model_parameters = merge_parameters(final_variables, fitting_parameters)  
    else:
        model_parameters = variables
    if more_output:
        # Simulate the PDS time traces
        simulated_time_traces, background_parameters, background_time_traces, background_free_time_traces, simulated_spectra, modulation_depths, dipolar_angle_distributions = \
            simulator.simulate_time_traces(experiments, spins, model_parameters, reset_field_orientations, more_output, display_messages=False)
        return simulated_time_traces, background_parameters, background_time_traces, background_free_time_traces, simulated_spectra, modulation_depths, dipolar_angle_distributions
    else:
        # Simulate PDS time traces
        simulated_time_traces = simulator.simulate_time_traces(experiments, spins, model_parameters, reset_field_orientations, more_output, display_messages=False)
        return simulated_time_traces


def objective_function(variables, simulator, experiments, spins, fitting_parameters, goodness_of_fit, reset_field_orientations, fixed_variables_included):
    ''' Objective function '''
    # Compute the fit
    simulated_time_traces = fit_function(variables, simulator, experiments, spins, fitting_parameters, reset_field_orientations, fixed_variables_included, more_output=False)
    # Compute the score 
    if goodness_of_fit == 'chi2':
        score = 0
        for i in range(len(experiments)):
            score += chi2(simulated_time_traces[i]['s'], experiments[i].s, experiments[i].noise_std) 
        return score

    
def objective_function_with_background_record(variables, simulator, experiments, spins, fitting_parameters, goodness_of_fit, reset_field_orientations, fixed_variables_included):
    ''' Objective function with the record of the background parameters '''
    # Compute the fit
    simulated_time_traces, background_parameters, background_time_traces, background_free_time_traces, simulated_spectra, modulation_depths, dipolar_angle_distributions = \
        fit_function(variables, simulator, experiments, spins, fitting_parameters, reset_field_orientations, fixed_variables_included, more_output=True)
    # Compute the score
    if goodness_of_fit == 'chi2':
        score = 0
        for i in range(len(experiments)):   
            score += chi2(simulated_time_traces[i]['s'], experiments[i].s, experiments[i].noise_std) 
        return score, background_parameters, modulation_depths