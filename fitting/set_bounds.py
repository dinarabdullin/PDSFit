from supplement.definitions import const


def set_bounds(fitting_parameters):
    """Set bounds for model parameters that will be optimized."""
    bounds = []
    for parameter_name in const["model_parameter_names"]:
        for fitting_parameter in fitting_parameters[parameter_name]:
            if fitting_parameter.optimized:
                bounds.append(fitting_parameter.get_range())
    return bounds 