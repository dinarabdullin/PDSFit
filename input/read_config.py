import os
import sys
import io
import libconf
import numpy as np
sys.path.append("..")
from input.libconf2data import libconf2data
from experiments.experiment_types import experiment_types
from spin_system.spin import Spin
from background.background_types import background_types
from simulation.simulator_types import simulator_types
from fitting.parameter_id import ParameterID
from fitting.optimization_methods import optimization_methods
from fitting.scoring_function import goodness_of_fit_parameters
from error_analysis.error_analyzer import ErrorAnalyzer
from output.data_saver import DataSaver
from output.logger import Logger
from plots.plotter import Plotter
from supplement.definitions import const


def read_output_settings(config, filepath_config):
    """Read out the output settings."""
    save_data = bool(config["output"]["save_data"])
    save_figures = bool(config["output"]["save_figures"])
    output_directory = config["output"]["directory"]
    data_saver = DataSaver(save_data, save_figures)
    data_saver.create_output_directory(output_directory, filepath_config)
    if data_saver.directory != "":
        sys.stdout = Logger(data_saver.directory + "logfile.log")
    return data_saver


def read_calculation_mode(config):
    """Read out the operation mode of PDSFit."""
    switch = int(config["mode"])
    if switch == 0:
        mode = {"simulation": 1, "fitting": 0, "error_analysis": 0}
    elif switch == 1:
        mode = {"simulation": 0, "fitting": 1, "error_analysis": 1}
    elif switch == 2:
        mode = {"simulation": 0, "fitting": 0, "error_analysis": 1}
    else:
        raise ValueError("Invalid operation mode!")
        sys.exit(1)
    return mode


def read_experimental_parameters(config, mode):
    """Read out PDS time traces and corresponding experimental parameters."""
    experiments = []
    for instance in config["experiments"]:
        experiment_name = instance["name"]
        technique = instance["technique"] 
        if technique in experiment_types:
            experiment = experiment_types[technique](experiment_name)
            parameter_values = {}
            for parameter_name, data_type in experiment.parameter_names.items():
                parameter_values[parameter_name] = libconf2data(instance[parameter_name], data_type=data_type)
            experiment.set_parameters(parameter_values)
            experiment.load_signal_from_file(instance.filename)
            if "phase" in instance:
                phase = float(instance["phase"])
            else:
                phase = np.nan
            if "zero_point" in instance:
                zero_point = float(instance["zero_point"])
            else:
                zero_point = np.nan
            if "noise_std" in instance:
                noise_std = float(instance["noise_std"])
            else:
                noise_std = np.nan
            experiment.perform_preprocessing(phase, zero_point, noise_std)
            sys.stdout.write("\nExperiment \'{0}\' was loaded\n".format(experiment_name))
            sys.stdout.write("Phase correction: {0:.0f} deg\n".format(experiment.phase))
            sys.stdout.write("Zero point: {0:.3f} us\n".format(experiment.zero_point))
            sys.stdout.write("Noise std: {0:0.6f}\n".format(experiment.noise_std))
            sys.stdout.flush()
            experiments.append(experiment)           
        else:
            raise ValueError("Invalid type of experiment!")
            sys.exit(1)  
    if experiments == []:
        raise ValueError("At least one experiment has to be provided!")
        sys.exit(1)
    return experiments
    

def read_spin_parameters(config):
    """Read out the parameters of a spin system."""
    spins = []
    for instance in config["spins"]:
        spin = Spin()
        parameter_values = {}
        for parameter_name, data_type in spin.parameter_names.items():
            if parameter_name in instance:
                parameter_values[parameter_name] = libconf2data(instance[parameter_name], data_type=data_type)
            else:
                parameter_values[parameter_name] = const["default_epr_parameters"][parameter_name]
        spin.set_parameters(parameter_values)  
        spins.append(spin)
    if len(spins) != 2:
        raise ValueError("Invalid number of spins!")
        sys.exit(1)
    return spins


def read_background_parameters(config):
    """Read out the parameters of a PDS background model."""
    model = config["background"]["model"]
    if model in background_types:
        background_model = (background_types[model])()
        parameters = {}
        for parameter_name, data_type in background_model.parameter_names.items():
            parameters[parameter_name] = {}
            parameters[parameter_name]["optimize"] = bool(config["background"]["parameters"][parameter_name]["optimize"])
            if parameters[parameter_name]["optimize"]:
                parameters[parameter_name]["range"] = libconf2data(config["background"]["parameters"][parameter_name]["range"], data_type=data_type)
            else:
                parameters[parameter_name]["range"] = []
            parameters[parameter_name]["value"] = libconf2data(config["background"]["parameters"][parameter_name]["value"], data_type=data_type)
        background_model.set_parameters(parameters)
    else:   
        raise ValueError("Invalid background model!")
        sys.exit(1)
    return background_model


def read_simulation_parameters(config):
    """Read out the simulation parameters."""
    simulation_parameters = {}
    for parameter_name in const["model_parameter_names"]:
        val = libconf2data(config["simulation_parameters"][parameter_name], data_type="float")
        if isinstance(val, float):
            val = [val]
        if len(val) == 0:
            raise ValueError("At least one value must be provided for \'{0}\'!".format(parameter_name))
        simulation_parameters[parameter_name] = [const["model_parameter_scales"][parameter_name] * v for v in val]
    for item in ["r_width", "xi_mean", "xi_width", "phi_mean", "phi_width", "alpha_mean", "alpha_width", \
        "beta_mean", "beta_width", "gamma_mean", "gamma_width", "j_mean", "j_width"]:
        if len(simulation_parameters["r_width"]) != len(simulation_parameters[item]):
            raise ValueError("Parameters \'{0}\' and \'{1}\' must have same dimensions!".format("r_mean", item))
            sys.exit(1)      
    return simulation_parameters


def read_fitting_parameters(config):
    """Read out the fitting parameters."""
    fitting_parameters = {}
    components = {}
    fitting_index = 0
    for parameter_name in const["model_parameter_names"]:
        instance = config["fitting_parameters"][parameter_name]
        list_optimize = libconf2data(instance["optimize"], data_type="int")
        list_range = libconf2data(instance["range"], data_type="float")
        list_value = libconf2data(instance["value"], data_type="float")
        if len(list_optimize) == 0:
            raise ValueError("At least one parameter must correspond to \'{0}\'!".format(parameter_name))
        else:
            fitting_parameters[parameter_name] = []
            components[parameter_name] = len(list_optimize)
            for i in range(len(list_optimize)):
                fitting_parameter = ParameterID(parameter_name, i)
                if list_optimize[i] == 1:
                    fitting_parameter.set_optimized(True)
                    fitting_parameter.set_index(fitting_index)
                    fitting_index += 1
                    fitting_parameter.set_range([const["model_parameter_scales"][parameter_name] * v for v in list_range[i]])
                elif list_optimize[i] == 0:
                    fitting_parameter.set_optimized(False)
                    fitting_parameter.set_value(const["model_parameter_scales"][parameter_name] * list_value[i])
                fitting_parameters[parameter_name].append(fitting_parameter)
    for item in ["r_width", "xi_mean", "xi_width", "phi_mean", "phi_width", "alpha_mean", "alpha_width", \
        "beta_mean", "beta_width", "gamma_mean", "gamma_width", "j_mean", "j_width"]:
        if components["r_mean"] != components[item]:
            raise ValueError("Parameters \'{0}\' and \'{1}\' must have same dimensions!".format("r_mean", item))
            sys.exit(1)
    return fitting_parameters
    

def read_fitting_settings(config, experiments):
    """Read out the technical parameters of the fitting procedure."""
    goodness_of_fit = config["fitting_settings"]["goodness_of_fit"]
    if goodness_of_fit in goodness_of_fit_parameters:
        method = config["fitting_settings"]["optimization_method"]
        if method in optimization_methods:
            optimizer = optimization_methods[method](method)
            optimizer.set_goodness_of_fit(goodness_of_fit)
            instance = config["fitting_settings"]["parameters"]
            parameter_values = {}
            for parameter_name, data_type in optimizer.intrinsic_parameter_names.items():
                parameter_values[parameter_name] = libconf2data(instance[parameter_name], data_type = data_type)
            optimizer.set_intrinsic_parameters(parameter_values)
        else:
            raise ValueError("Invalid optimization method!")
            sys.exit(1)
    else:
        raise ValueError("Invalid goodness-of-fit!")
        sys.exit(1)
    return optimizer


def read_error_analysis_parameters(config, fitting_parameters):
    """Read out the parameters of the error analysis and their ranges."""
    error_analysis_parameters = []
    for instance in config["error_analysis_parameters"]:
        list_names = libconf2data(instance["names"])
        list_components = libconf2data(instance["components"], data_type = "int")
        list_ranges = libconf2data(instance["ranges"], data_type = "float")
        if len(list_names) == 0:
            pass
        else:
            subset_parameters = []
            for i, parameter_name in enumerate(list_names): 
                if len(list_components) == 0:
                    component = 0
                else:
                    component = list_components[i] - 1
                error_analysis_parameter = ParameterID(parameter_name, component)
                found = False
                for fitting_parameter in fitting_parameters[parameter_name]:
                    if error_analysis_parameter == fitting_parameter and fitting_parameter.is_optimized():
                        error_analysis_parameter.set_index(fitting_parameter.get_index())
                        found = True
                        break
                if not found: 
                    raise ValueError("All error analysis parameters must be the fitting parameters!")
                    sys.exit(1)
                if len(list_ranges) == 0:
                    opt_range = fitting_parameter.get_range()
                    if opt_range is None:
                        raise ValueError("No range was provided for error analysis parameter \'{0}\'!".format(parameter_name))
                        sys.exit(1)
                    else:
                        if len(opt_range) == 0:
                            raise ValueError("No range was provided for error analysis parameter \'{0}\'!".format(parameter_name))
                            sys.exit(1)
                        else:
                            error_analysis_parameter.set_range(opt_range)
                else:
                    error_analysis_parameter.set_range([const["model_parameter_scales"][parameter_name] * v for v in list_ranges[i]])
                subset_parameters.append(error_analysis_parameter)      
            error_analysis_parameters.append(subset_parameters)
    return error_analysis_parameters


def read_error_analysis_settings(config, mode):
    """Read out the error analysis settings."""
    error_analyzer = ErrorAnalyzer()
    parameter_values = {}
    for parameter_name, data_type in error_analyzer.intrinsic_parameter_names.items():
        parameter_values[parameter_name] = libconf2data(config["error_analysis_settings"][parameter_name], data_type=data_type)
    error_analyzer.set_intrinsic_parameters(parameter_values)
    return error_analyzer


def read_calculation_settings(config):
    """Read out calculation settings."""  
    integration_method = config["calculation_settings"]["integration_method"]
    if integration_method in simulator_types:
        simulator = (simulator_types[integration_method])()
        parameter_values = {}
        for parameter_name, data_type in simulator.intrinsic_parameter_names.items():
            parameter_values[parameter_name] = libconf2data(config["calculation_settings"][parameter_name], data_type=data_type)
        distribution_types = {}
        for parameter_name in ["r", "xi", "phi", "alpha", "beta", "gamma", "j"]:
            distribution_types[parameter_name] = config["calculation_settings"]["distribution_types"][parameter_name]
        for key in distribution_types:
            if not distribution_types[key] in const["distribution_types"]:
                raise ValueError("Unsupported type of distribution is encountered for %s!" % (key))
                sys.exit(1)
        parameter_values["distribution_types"] = distribution_types
        simulator.set_intrinsic_parameters(parameter_values)
    return simulator

  
def read_config(filepath):
    """Read input data from a configuration file."""
    sys.stdout.write(
        "\n########################################################################\
        \n# Reading the configuration file and preprocessing the PDS time traces #\
        \n########################################################################\n"
        )
    sys.stdout.flush()
    simulator = None
    simulation_parameters = {}
    optimizer = None
    fitting_parameters = {}
    error_analyzer = None
    error_analysis_parameters = {}
    with io.open(filepath) as input_file:
        config = libconf.load(input_file)
        data_saver = read_output_settings(config, filepath)
        mode = read_calculation_mode(config)
        experiments = read_experimental_parameters(config, mode)
        spins = read_spin_parameters(config)
        background_model = read_background_parameters(config)
        simulator = read_calculation_settings(config)
        simulator.set_background_model(background_model)
        if mode["simulation"]:
            simulation_parameters = read_simulation_parameters(config)
        if mode["fitting"]:
            fitting_parameters = read_fitting_parameters(config)
            optimizer = read_fitting_settings(config, experiments)
        if mode["error_analysis"]:
            error_analyzer = read_error_analysis_settings(config, mode)
            if not mode["fitting"]:
                fitting_parameters = error_analyzer.load_fitting_parameters()
            error_analysis_parameters = read_error_analysis_parameters(config, fitting_parameters)
            
        plotter = Plotter(data_saver)  
    input_data = {
        "mode":                         mode,
        "experiments":                  experiments,
        "spins":                        spins,
        "simulator":                    simulator,
        "simulation_parameters":        simulation_parameters,
        "optimizer":                    optimizer,
        "fitting_parameters":           fitting_parameters,
        "error_analyzer":               error_analyzer,
        "error_analysis_parameters":    error_analysis_parameters,
        "data_saver":                   data_saver,
        "plotter":                      plotter,
    }
    return input_data