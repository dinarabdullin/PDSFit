def compute_number_of_fitting_parameters(optimized_model_parameters, optimized_background_parameters):
    ''' Computes the number of fitting parameters ''' 
    num_model_parameters = len(optimized_model_parameters)
    num_background_parameters = 0
    for background_parameters_single_time_trace in optimized_background_parameters:
        num_background_parameters += len(background_parameters_single_time_trace)
    num_parameters = num_model_parameters + num_background_parameters
    return num_parameters, num_model_parameters, num_background_parameters