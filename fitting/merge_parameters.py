def merge_parameters(optimized_parameters, fitting_parameters):
    ''' Merges the optimized and fixed parameters of the model into a single dictionary '''
    # Merge the parameters
    merged_parameters = {}
    for parameter_name in fitting_parameters['indices']:
        single_parameter_indices = fitting_parameters['indices'][parameter_name]
        single_parameter_values = []
        for i in range(len(single_parameter_indices)):
            parameter_object = single_parameter_indices[i]
            if parameter_object.optimize:
                parameter_value = optimized_parameters[parameter_object.index]
            else:
                parameter_value = fitting_parameters['values'][parameter_object.index]
            single_parameter_values.append(parameter_value)
        merged_parameters[parameter_name] = single_parameter_values
    return merged_parameters