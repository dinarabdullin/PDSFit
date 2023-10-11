import numpy as np
from supplement.definitions import const


def save_symmetry_related_models(filepath, symmetry_related_models, fitting_parameters):    
    """Save symmetry-related geometric models of a spin system.""" 
    num_models = len(symmetry_related_models)
    file = open(filepath, "w")
    file.write("{:<20}".format("Parameter"))
    file.write("{:<15}".format("No. component"))
    for k in range(num_models):
        file.write("{:<15}".format(symmetry_related_models[k]["transformation"]))
    file.write("\n")
    for name in const["model_parameter_names"]:
        num_modes = len(symmetry_related_models[0]["parameters"][name])
        for i in range(num_modes):
            file.write("{:<20}".format(const["model_parameter_names_and_units"][name]))
            file.write("{:<15}".format(i + 1))
            for k in range(num_models):
                value = symmetry_related_models[k]["parameters"][name][i] / const["model_parameter_scales"][name]
                if name in const["angle_parameter_names"]:
                    file.write("{:<15.1f}".format(value))
                else:
                    file.write("{:<15.3f}".format(value))
            file.write("\n")
    file.write("\n")
    file.write("{:<35}".format("score"))
    for k in range(num_models):
        score_value = symmetry_related_models[k]["score"]
        file.write("{:<15.1f}".format(score_value))
    file.write("\n")
    file.close()