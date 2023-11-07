import sys
import time
import datetime
import numpy as np
import scipy
from scipy import interpolate
from copy import deepcopy
from multiprocessing import Pool
try:
    import mpi4py
    from mpi4py.futures import MPIPoolExecutor
except:
    pass
from mpi.mpi_status import get_mpi
from error_analysis.load_optimized_models import load_optimized_models, load_fitting_parameters
from fitting.scoring_function import normalize_weights
from mathematics.find_nearest import find_nearest


class ErrorAnalyzer():
    """Error analysis."""

    def __init__(self):
        self.scoring_function = None
        self.intrinsic_parameter_names = {
            "confidence_interval": "int", 
            "samples_per_parameter": "int",
            "samples_numerical_error": "int",
            "filepath_fitting_parameters": "str"
            }
    
    
    def set_intrinsic_parameters(self, intrinsic_parameters):
        """Set the intrinsic parameters."""
        self.confidence_interval = intrinsic_parameters["confidence_interval"]
        self.samples_per_parameter = intrinsic_parameters["samples_per_parameter"]
        self.samples_numerical_error = intrinsic_parameters["samples_numerical_error"]
        self.filepath_fitting_parameters = intrinsic_parameters["filepath_fitting_parameters"]
    
    
    def load_fitting_parameters(self):
        if self.filepath_fitting_parameters:
            return load_fitting_parameters(self.filepath_fitting_parameters)
        else:
            raise ValueError("A file with an optimized model is missing!")
            sys.exit(1)
    
    
    def load_optimized_models(self):
        """Load the optimized values of the model parameters from a file."""
        if self.filepath_fitting_parameters:
            return load_optimized_models(self.filepath_fitting_parameters)
        else:
            raise ValueError("A file with an optimized model is missing!")
            sys.exit(1)    
    
    
    def set_scoring_function(self, func):
        """Set the objective function."""
        self.scoring_function = func


    def run_error_analysis(
        self, error_analysis_parameters, optimized_model_parameters, simulated_data_optimized_model,
        fitting_parameters, background_model, experiments 
        ):    
        """Run the error analysis."""
        sys.stdout.write(
            "\n########################################################################\
            \n#                            Error analysis                            #\
            \n########################################################################\n"
            )
        sys.stdout.flush()
        time_start = time.time()
        sys.stdout.write("\nComputing the chi-squared threshold...\n")
        sys.stdout.flush()
        chi2_thresholds, chi2_minimum = self.compute_chi2_threshold(error_analysis_parameters, optimized_model_parameters)
        sys.stdout.write("\nComputing the errors of the fitting parameters...\n")
        sys.stdout.flush()
        all_error_surfaces, all_error_surfaces_2d, all_error_surfaces_1d = [], [], []
        errors_model_parameters = self.init_errors_model_parameters(optimized_model_parameters)
        errors_background_parameters = self.init_errors_background_parameters(simulated_data_optimized_model["background_parameters"], background_model)
        errors_backgrounds = self.init_errors_backgrounds(simulated_data_optimized_model["background"])
        num_parameter_subspaces = len(error_analysis_parameters)
        for i in range(num_parameter_subspaces):
            sys.stdout.write("Parameter set {0} / {1}\n".format(i + 1, num_parameter_subspaces))
            sys.stdout.flush()
            parameter_subspace = error_analysis_parameters[i]
            num_parameters = len(parameter_subspace)
            # Compute an error surface
            error_surface, simulated_data_error_surface = self.compute_error_surface(parameter_subspace, optimized_model_parameters, fitting_parameters)
            all_error_surfaces.append(error_surface)
            # If the dimension of the error surface is larger than two, project 
            # the the error surface onto two-dimensional parameter subspaces.
            if num_parameters > 2:
                error_surfaces_2d = self.compute_2d_error_surfaces(error_surface)
                all_error_surfaces_2d.extend(error_surfaces_2d)
            # Project the the error surface onto one-dimensional parameter subspaces.
            if num_parameters > 1:
                error_surfaces_1d = self.compute_1d_error_surfaces(error_surface)
            else:
                error_surfaces_1d = [error_surface]
            for error_surface_1d in error_surfaces_1d:
                 error_surface_1d = self.reset_minimum_chi2(error_surface_1d, chi2_minimum, optimized_model_parameters)
            all_error_surfaces_1d.extend(error_surfaces_1d)
            # Compute the errors of model parameters
            for error_surface_1d in error_surfaces_1d:
                error_model_parameter = self.compute_model_parameter_error(optimized_model_parameters, error_surface_1d, chi2_thresholds, chi2_minimum)
                errors_model_parameters = self.update_errors_model_parameters(error_surface_1d["par"][0], error_model_parameter, errors_model_parameters)
            # Compute the backgound errors
            new_errors_background_parameters, new_errors_backgrounds = self.compute_background_errors(
                simulated_data_optimized_model["background_parameters"], simulated_data_optimized_model["background"], 
                error_surface, simulated_data_error_surface, chi2_thresholds, chi2_minimum, background_model, experiments
                )
            errors_background_parameters = self.update_errors_background_parameters(new_errors_background_parameters, errors_background_parameters)
            errors_backgrounds = self.update_errors_backgrounds(new_errors_backgrounds, errors_backgrounds)           
        errors_distributions = self.compute_errors_distributions(optimized_model_parameters, errors_model_parameters, fitting_parameters)
        return {
            "error_surfaces": all_error_surfaces,
            "error_surfaces_2d": all_error_surfaces_2d,
            "error_surfaces_1d": all_error_surfaces_1d,
            "chi2_minimum": chi2_minimum,
            "chi2_thresholds": chi2_thresholds,
            "errors_model_parameters": errors_model_parameters,
            "errors_background_parameters" : errors_background_parameters,
            "errors_backgrounds": errors_backgrounds
            }
    
    
    def compute_chi2_threshold(self, error_analysis_parameters, optimized_model_parameters):
        """Compute the chi-squared threshold."""
        # Compute theoretical values of the chi2 threshold for 
        # various degrees of freedom at a fixed confidence level.
        num_parameters = len(optimized_model_parameters)
        max_dim = 0
        for i in range(len(error_analysis_parameters)):
            dim = len(error_analysis_parameters[i])
            if dim > max_dim:
                max_dim = dim
        degrees_of_freedom = np.arange(num_parameters, num_parameters - max_dim, -1)
        chi2_thresholds_theory = self.compute_theoretical_chi2_thresholds(degrees_of_freedom, self.confidence_interval)
        # Estimate the contribution of the numerical error to the chi2 threshold
        mean_chi2_minimum, std_chi2_minimum = self.compute_numerical_error(optimized_model_parameters)
        chi2_threshold_num_error = float(self.confidence_interval) * std_chi2_minimum
        # Compute the total chi2 threshold(s)
        total_chi2_thresholds = chi2_thresholds_theory + chi2_threshold_num_error 
        # Print the chi2 threshold(s)
        sys.stdout.write("Minimum chi-squared: {0:<.1f}\n".format(mean_chi2_minimum))
        sys.stdout.write(
            "Theoretical chi-squared threshold ({0:d}-sigma): ".format(self.confidence_interval)
            )
        for i in range(max_dim):
            sys.stdout.write("{0:<.1f} ({1:d}d)".format(chi2_thresholds_theory[i], i + 1))
            if i < max_dim - 1:
                sys.stdout.write(", ")
            else:
                sys.stdout.write("\n")
        sys.stdout.write(
            "Numerical error contribution ({0:d}-sigma): {1:<.1f}\n".format(self.confidence_interval, chi2_threshold_num_error)
            )
        sys.stdout.write(
            "Total chi-squared threshold ({0:d}-sigma): ".format(self.confidence_interval)
            )
        for i in range(max_dim):
            sys.stdout.write("{0:<.1f} ({1:d}d)".format(total_chi2_thresholds[i], i + 1))
            if i < max_dim - 1:
                sys.stdout.write(", ")
            else:
                sys.stdout.write("\n")
        return total_chi2_thresholds, mean_chi2_minimum
    
    
    def compute_theoretical_chi2_thresholds(self, degrees_of_freedom, confidence_interval):
        """Compute theoretical values of the chi-squared threshold for
        various degrees of freedom at a fixed confidence level."""
        chi2_thresholds_theory = []
        for v in degrees_of_freedom:
            chi2_threshold = 0.0
            if v == 1:
                chi2_threshold = float(confidence_interval)**2
            else:
                p = 1.0 - scipy.stats.chi2.sf(float(confidence_interval)**2, 1)
                chi2_threshold = scipy.stats.chi2.ppf(p, int(v))
            chi2_thresholds_theory.append(chi2_threshold)
        return np.array(chi2_thresholds_theory)
    
    
    def compute_numerical_error(self, optimized_model_parameters):
        """Compute the contribution of the numerical error to the chi-squared treshold."""
        # Make multiple copies of the optimized model parameters
        parameter_sets = []
        for i in range(self.samples_numerical_error):
            parameter_sets.append(optimized_model_parameters)
        # Calculate chi-squared values
        run_with_mpi = get_mpi()
        if run_with_mpi:
            with MPIPoolExecutor() as executor:
                result = zip(*executor.map(self.scoring_function, parameter_sets))
            chi2_values, _ = list(result)
        else:
            pool = Pool()
            chi2_values, _ = list(zip(*pool.map(self.scoring_function, parameter_sets)))
            pool.close()
            pool.join()
        chi2_values = np.array(chi2_values)
        # Set the minimum chi2 and the chi2 threshold due to the numerical error
        mean_chi2_minimum, std_chi2_minimum = np.mean(chi2_values), np.std(chi2_values)
        # Plot
        # from plots.error_analysis.plot_numerical_error import plot_numerical_error
        # plot_numerical_error(chi2_values, mean_chi2_minimum, std_chi2_minimum)
        return mean_chi2_minimum, std_chi2_minimum

    
    def compute_error_surface(self, parameter_subspace, optimized_model_parameters, fitting_parameters):
        """Compute an error surface."""
        num_parameters = len(parameter_subspace)
        # Generate model parameters set with different values for error analysis parameters
        if num_parameters == 1:
            num_samples = self.samples_per_parameter
            parameter_sets = np.tile(optimized_model_parameters, (num_samples, 1))
            parameter = parameter_subspace[0]
            parameter_index = parameter.get_index()
            parameter_range = parameter.get_range()
            parameter_values = np.linspace(parameter_range[0], parameter_range[1], num = num_samples)
            parameter_grid_points = np.expand_dims(parameter_values, -1)
            parameter_sets[:,parameter_index] = parameter_grid_points[:,0]
        else:
            num_samples = np.power(self.samples_per_parameter, num_parameters)
            parameter_sets = np.tile(optimized_model_parameters, (num_samples, 1))
            parameter_axes, parameter_indices = [], []
            for j in range(num_parameters):
                parameter = parameter_subspace[j]
                parameter_index = parameter.get_index()
                parameter_range = parameter.get_range()
                parameter_values = np.linspace(parameter_range[0], parameter_range[1], num = self.samples_per_parameter)
                parameter_axes.append(parameter_values)
                parameter_indices.append(parameter_index)
            parameter_grid_points = np.stack(np.meshgrid(*parameter_axes, indexing="ij"), -1).reshape(num_samples, num_parameters)
            # parameter_grid = np.reshape(np.transpose(parameter_grid_points), [num_parameters] + [self.samples_per_parameter] * num_parameters)
            # print(parameter_grid.shape)
            # assert np.all(parameter_grid == np.array(np.meshgrid(*parameter_axes, indexing="ij")))
            for j in range(num_parameters):
                parameter_sets[:,parameter_indices[j]] = parameter_grid_points[:,j]
        # Normalize relative weights
        for k in range(num_samples):
            parameter_sets[k,:] = normalize_weights(parameter_sets[k,:], fitting_parameters)
        # Calculate chi-squared values
        run_with_mpi = get_mpi()
        if run_with_mpi:
            with MPIPoolExecutor() as executor:
                result = zip(*executor.map(self.scoring_function, parameter_sets))
            chi2_values, simulated_data = list(result)
        else:
            pool = Pool()
            chi2_values, simulated_data = list(zip(*pool.map(self.scoring_function, parameter_sets)))
            pool.close()
            pool.join()
        chi2_values, simulated_data = np.array(chi2_values), np.array(simulated_data)
        # chi2_grid = np.reshape(chi2_values, [self.samples_per_parameter] * num_parameters)
        # Store error analysis
        error_surface = {}
        error_surface["par"] = parameter_subspace
        error_surface["x"] = np.transpose(parameter_grid_points)
        error_surface["y"] = chi2_values
        return error_surface, simulated_data       

    
    def compute_2d_error_surfaces(self, error_surface):
        """Project an n-dimentional error surface (n > 2) 
        onto two-dimentional parameter subspaces."""
        # Convert parameters' values and chi-squared values to
        # a parameters' grid and a chi-squared grid, respectively.
        parameters, parameter_grid_points, chi2_values = error_surface["par"], error_surface["x"], error_surface["y"]
        num_parameters = len(parameters)
        parameter_grid = np.reshape(parameter_grid_points, [num_parameters] + [self.samples_per_parameter] * num_parameters)
        chi2_grid = np.reshape(chi2_values, [self.samples_per_parameter] * num_parameters)
        # Compute two-dimensional error surfaces
        error_surfaces_2d = []
        for i in range(num_parameters - 1):
            for j in range(i + 1, num_parameters):
                # Transpose the parameters' grid such that the two variables of an error surface 
                # will correspond to last two axes of the the parameters' grid.
                new_index_order = []
                for k in range(num_parameters):
                    if k != i and k != j:
                        new_index_order += [k]
                new_index_order += [i, j]
                print(new_index_order)
                parameter1_grid = parameter_grid[i]
                parameter2_grid = parameter_grid[j]
                parameter1_grid = np.transpose(parameter1_grid, axes = new_index_order)
                parameter2_grid = np.transpose(parameter2_grid, axes = new_index_order)
                new_chi2_grid = np.transpose(chi2_grid, axes = new_index_order)
                # Reduce the dimension of the parameters' and chi-squared grids to 2
                for _ in range(num_parameters-2):
                    parameter1_grid = parameter1_grid[0]
                    parameter2_grid = parameter2_grid[0]
                    new_chi2_grid = np.amin(new_chi2_grid, axis = 0)
                new_parameter_grid = np.array([parameter1_grid, parameter2_grid])
                new_parameter_grid_points = np.stack(new_parameter_grid, -1).reshape(self.samples_per_parameter**2, 2)
                new_chi2_grid = np.expand_dims(new_chi2_grid, 0)
                new_chi2_values = new_chi2_grid.reshape(-1, self.samples_per_parameter**2)
                error_surface_2d = {}  
                error_surface_2d["par"] = [parameters[i], parameters[j]]
                error_surface_2d["x"] = np.transpose(new_parameter_grid_points)
                error_surface_2d["y"] = new_chi2_values[0]
                error_surfaces_2d.append(error_surface_2d)
        return error_surfaces_2d
    
    
    def compute_1d_error_surfaces(self, error_surface):
        """Project an n-dimentional error surface (n > 2) 
        onto one-dimentional parameter subspaces."""
        # Convert parameters' values and chi-squared values to
        # a parameters' grid and a chi-squared grid, respectively.
        parameters, parameter_grid_points, chi2_values = error_surface["par"], error_surface["x"], error_surface["y"]
        num_parameters = parameter_grid_points.shape[0]
        parameter_grid = np.reshape(parameter_grid_points, [num_parameters] + [self.samples_per_parameter] * num_parameters)
        chi2_grid = np.reshape(chi2_values, [self.samples_per_parameter] * num_parameters)
        # Compute two-dimensional error surfaces
        error_surfaces_1d = []
        for i in range(num_parameters):
            # Transpose the parameters' grid such that the variable of an error surface 
            # will correspond the last axis of the the parameters' grid.
            new_index_order = []
            for k in range(num_parameters):
                if k != i:
                    new_index_order += [k]
            new_index_order += [i]
            parameter1_grid = parameter_grid[i]
            parameter1_grid = np.transpose(parameter1_grid, axes = new_index_order)
            new_chi2_grid = np.transpose(chi2_grid, axes = new_index_order)
            # Reduce the dimension of the parameters' and chi-squared grids to 2
            for _ in range(num_parameters-1):
                parameter1_grid = parameter1_grid[0]
                new_chi2_grid = np.amin(new_chi2_grid, axis = 0)
            parameter1_grid_points = np.expand_dims(parameter1_grid, -1)
            error_surface_1d = {}  
            error_surface_1d["par"] = [parameters[i]]
            error_surface_1d["x"] = np.transpose(parameter1_grid_points)
            error_surface_1d["y"] = new_chi2_grid
            error_surfaces_1d.append(error_surface_1d)
        return error_surfaces_1d
    
    
    def reset_minimum_chi2(self, error_surface_1d, chi2_minimum, optimized_model_parameters):
        """Reset the minimum chi-squared value of an one-dimensional error surface."""
        parameter, parameter_values, chi2_values = error_surface_1d["par"][0], error_surface_1d["x"][0], error_surface_1d["y"] 
        optimized_value = optimized_model_parameters[parameter.get_index()]
        index_optimized_value = find_nearest(parameter_values, optimized_value)
        if index_optimized_value <= 1:
            current_chi2_minimum = np.mean(chi2_values[0:5])
        elif index_optimized_value == self.samples_per_parameter - 2:
            current_chi2_minimum = np.mean(chi2_values[-5:])
        else:
            current_chi2_minimum = np.mean(chi2_values[index_optimized_value-2:index_optimized_value+2])
        if current_chi2_minimum < chi2_minimum:
            new_chi2_values = chi2_values - current_chi2_minimum + chi2_minimum
            error_surface_1d["y"] = new_chi2_values
        return error_surface_1d
    
    
    def init_errors_model_parameters(self, optimized_model_parameters):
        """Initialize background errors."""
        return [[np.nan, np.nan]] * len(optimized_model_parameters)


    def update_errors_model_parameters(self, parameter, error, errors_model_parameters):
        """Update the errors of model parameters."""
        if error != [np.nan, np.nan]:
            parameter_index = parameter.get_index()
            if errors_model_parameters[parameter_index] == [np.nan, np.nan]:
                errors_model_parameters[parameter_index] = error
            else:
                if error[0] < errors_model_parameters[parameter_index][0]:
                    errors_model_parameters[parameter_index][0] = error[0]
                if error[1] > errors_model_parameters[parameter_index][1]:
                    errors_model_parameters[parameter_index][1] = error[1]
        return errors_model_parameters    
    
    
    def compute_model_parameter_error(
        self, optimized_model_parameters, error_surface_1d, chi2_thresholds, chi2_minimum, delta = 4 
        ):
        """Compute the error of an optimized model parameter."""
        parameter, parameter_values, chi2_values = error_surface_1d["par"][0], error_surface_1d["x"][0], error_surface_1d["y"] 
        optimized_value = optimized_model_parameters[parameter.get_index()]
        min_value, max_value = np.amin(parameter_values), np.amax(parameter_values)
        step = parameter_values[1] - parameter_values[0]
        if parameter_values.size < 100:
            parameter_grid = np.linspace(min_value, max_value, 100)
            interpolation = interpolate.interp1d(parameter_values, chi2_values, kind = "cubic")
            chi2_grid = interpolation(parameter_grid)
        else:
            parameter_grid = parameter_values
            chi2_grid = chi2_values
        # Find the parameter values below the threshold
        indices_uncertainty_interval = np.where(chi2_grid <= chi2_minimum + chi2_thresholds[0])[0]
        error = [np.nan, np.nan]
        if indices_uncertainty_interval.size <= 1:
            sys.stdout.write(
                "WARNING: The uncertanty interval of parameter \'{0}\' is below the resolution of the error surface! ".format(parameter.name)
                )
            sys.stdout.write("Reduce the parameter range or increase the resolution of the error surface.\n")
            sys.stdout.flush()
        else:
            # Check whether there are several uncertainty intervals separated from each other
            parameter_values_uncertainty_interval = parameter_grid[indices_uncertainty_interval]
            lower_bounds, upper_bounds = [], []
            lower_bounds.append(parameter_values_uncertainty_interval[0])
            for i in range(1, parameter_values_uncertainty_interval.size - 1):
                if (parameter_values_uncertainty_interval[i] - parameter_values_uncertainty_interval[i-1]) > delta * step:
                    lower_bounds.append(parameter_values_uncertainty_interval[i])
                    upper_bounds.append(parameter_values_uncertainty_interval[i-1])        
            upper_bounds.append(parameter_values_uncertainty_interval[-1])        
            all_bounds = np.column_stack((lower_bounds, upper_bounds))   
            # Find the uncertainty interval that contains the optimized value of the model parameter
            bounds_uncertainty_interval = None
            for bounds in all_bounds:
                if (bounds[0] - step <= optimized_value) and (bounds[1] + step >= optimized_value):
                    bounds_uncertainty_interval = bounds
            if bounds_uncertainty_interval is None:
                sys.stdout.write(
                    "WARNING: The optimized value of parameter \"{0}\" is outside the calculated uncertanty interval!\n".format(parameter.name)
                    )
                sys.stdout.flush()
            else:  
                if bounds_uncertainty_interval[1] - bounds_uncertainty_interval[0] < step:
                    sys.stdout.write(
                        "WARNING: The uncertanty interval of parameter \'{0}\' is below the resolution of the error surface! ".format(parameter.name)
                        )
                    sys.stdout.write("Reduce the parameter range or increase the resolution of the error surface.\n")
                    sys.stdout.flush()
                elif (bounds_uncertainty_interval[0] <= min_value + step) and (bounds_uncertainty_interval[1] >= max_value - step):
                    sys.stdout.write(
                        "WARNING: The uncertanty interval of parameter \'{0}\' spans over its entire range!\n".format(parameter.name)
                        )
                    sys.stdout.flush()
                else:
                    error = [
                        bounds_uncertainty_interval[0] - optimized_value, 
                        bounds_uncertainty_interval[1] - optimized_value
                        ]
        return error
 
    
    def init_errors_background_parameters(self, optimized_background_parameters, background_model):
        """Initialize background errors."""
        errors_background_parameters = []
        for i in range(len(optimized_background_parameters)):
            errors_single_experiment = {}
            for name in optimized_background_parameters[i]:
                if background_model.parameters[name]["optimize"]:
                    errors_single_experiment[name] = [0.0, 0.0]
            errors_background_parameters.append(errors_single_experiment)
        return errors_background_parameters
    
    
    def update_errors_background_parameters(self, new_errors_background_parameters, current_errors_background_parameters):
        """Update background errors."""
        for i in range(len(current_errors_background_parameters)):
            for name in current_errors_background_parameters[i]:
                current_error = current_errors_background_parameters[i][name]
                new_error = new_errors_background_parameters[i][name]
                if new_error != [np.nan, np.nan]:
                    if current_error == [np.nan, np.nan]:
                        current_errors_background_parameters[i][name] = new_error
                    else:
                        if new_error[0] < current_error[0]:
                            current_errors_background_parameters[i][name][0] = new_error[0]
                        if new_error[1] > current_error[1]:
                            current_errors_background_parameters[i][name][1] = new_error[1]
        return current_errors_background_parameters
    
    
    def init_errors_backgrounds(self, optimized_backgrounds):
        """Initialize background errors."""
        errors_backgrounds = []
        for i in range(len(optimized_backgrounds)):
            errors_background = np.zeros((2, optimized_backgrounds[i].size))
            errors_backgrounds.append(errors_background)
        return errors_backgrounds
    
    
    def update_errors_backgrounds(self, new_errors_backgrounds, current_errors_backgrounds):
        """Update background errors."""
        for i in range(len(current_errors_backgrounds)):
            indices_lower_bound = np.where(new_errors_backgrounds[i][0] < current_errors_backgrounds[i][0])[0]
            if indices_lower_bound.size > 0:
                current_errors_backgrounds[i][0][indices_lower_bound] = new_errors_backgrounds[i][0][indices_lower_bound]
            indices_upper_bound = np.where(new_errors_backgrounds[i][1] > current_errors_backgrounds[i][1])[0]
            if indices_upper_bound.size > 0:
                current_errors_backgrounds[i][1][indices_upper_bound] = new_errors_backgrounds[i][1][indices_upper_bound]
        return current_errors_backgrounds
    
    
    def compute_background_errors(
        self, optimized_background_parameters, optimized_backgrounds, error_surface, simulated_data, 
        chi2_thresholds, chi2_minimum, background_model, experiments
        ):
        """Compute the errors of optimized background parameters and
        the corresponing errors of optimized backgrounds."""
        errors_background_parameters = self.init_errors_background_parameters(optimized_background_parameters, background_model)
        errors_backgrounds = self.init_errors_backgrounds(optimized_backgrounds)
        num_parameters = len(error_surface["x"])
        chi2_values = error_surface["y"]
        indices_uncertainty_interval = np.where(chi2_values <= chi2_minimum + chi2_thresholds[num_parameters - 1])[0]
        if indices_uncertainty_interval.size > 0:
            simulated_data_uncertainty_interval = simulated_data[indices_uncertainty_interval]
            # Compute the errors of optimized background parameters
            for i in range(len(optimized_background_parameters)):
                for name in optimized_background_parameters[i]:
                    if background_model.parameters[name]["optimize"]:
                        optimized_value = optimized_background_parameters[i][name]
                        parameter_values_uncertainty_interval = np.zeros(len(simulated_data_uncertainty_interval))
                        for k in range(len(simulated_data_uncertainty_interval)):
                            parameter_values_uncertainty_interval[k] = simulated_data_uncertainty_interval[k]["background_parameters"][i][name]
                        bounds_uncertainty_interval = [
                            np.amin(parameter_values_uncertainty_interval), 
                            np.amax(parameter_values_uncertainty_interval)
                            ]
                        error = [
                            bounds_uncertainty_interval[0] - optimized_value, 
                            bounds_uncertainty_interval[1] - optimized_value
                            ]
                        errors_background_parameters[i][name] = error
            # Compute the errors of optimized backgrounds
            for k in range(len(simulated_data_uncertainty_interval)):
                for i in range(len(experiments)):
                    background_parameters = simulated_data_uncertainty_interval[k]["background_parameters"][i]
                    modulation_depth = simulated_data_uncertainty_interval[k]["modulation_depth"][i]
                    background = background_model.get_background(experiments[i].t, background_parameters, modulation_depth)
                    residuals = background - optimized_backgrounds[i]
                    indices_lower_bound = np.where(residuals < errors_backgrounds[i][0])[0]
                    if indices_lower_bound.size > 0:
                        errors_backgrounds[i][0][indices_lower_bound] = residuals[indices_lower_bound]
                    indices_upper_bound = np.where(residuals > errors_backgrounds[i][1])[0]
                    if indices_upper_bound.size > 0:
                        errors_backgrounds[i][1][indices_upper_bound] = residuals[indices_upper_bound]
        return errors_background_parameters, errors_backgrounds
    
    
    def compute_errors_distributions(self, optimized_model_parameters, errors_model_parameters, fitting_parameters):
        """Compute error bars for the distributions of model parameters."""
        
        