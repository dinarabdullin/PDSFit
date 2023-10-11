import sys
import numpy as np
from copy import deepcopy
import itertools
from scipy.spatial.transform import Rotation
from scipy.optimize import curve_fit
from scipy.special import i0
from multiprocessing import Pool
try:
    import mpi4py
    from mpi4py.futures import MPIPoolExecutor
except:
    pass
from fitting.scoring_function import merge_optimized_and_fixed_parameters
from mathematics.coordinate_system_conversions import spherical2cartesian, cartesian2spherical 
from mathematics.random_samples_from_distribution import random_samples_from_distribution
from mathematics.histogram import histogram
from mpi.mpi_status import get_mpi
from supplement.definitions import const


def compute_symmetry_related_models(
    optimized_model, fitting_parameters, simulator, spins, scoring_function
    ):
    """Compute symmetry-related sets of fitting parameters."""
    sys.stdout.write("\nComputing the symmetry-related models... ")
    sys.stdout.flush()
    # Merge the optimized and fixed model parameters into a single dictionary
    model_parameters = merge_optimized_and_fixed_parameters(optimized_model, fitting_parameters)
    if spins[0] == spins[1]:
        # For the assignement: spin 1 = spin A, spin 2 = spin B
        spin_labels = ["A","B"]
        symmetry_related_models_set1 = compute_equivalent_models(
            model_parameters, fitting_parameters, simulator, scoring_function, spin_labels 
            )
        # For the assignement: spin 1 = spin B, spin 2 = spin A
        spin_labels = ["B","A"]
        model_parameters_set2 = compute_model_after_spin_exchange(
            model_parameters, fitting_parameters, simulator
            )
        symmetry_related_models_set2 = compute_equivalent_models(
            model_parameters_set2, fitting_parameters, simulator, scoring_function, spin_labels
            )
        symmetry_related_models = symmetry_related_models_set1 + symmetry_related_models_set2
    else:
        symmetry_related_models = compute_equivalent_models(
            model_parameters, fitting_parameters, simulator, scoring_function
            )
    sys.stdout.write("done!\n")
    sys.stdout.flush() 
    return symmetry_related_models


def compute_equivalent_models(
    model_parameters, fitting_parameters, simulator, scoring_function, spin_labels = ["A","B"]
    ):
    """Compute 16 sets of equivalent angles."""
    # Transformation matrices
    I = Rotation.from_euler("ZXZ", np.column_stack((0, 0, 0)))
    RX = Rotation.from_euler("ZXZ", np.column_stack((0, np.pi, 0))).inv()
    RY = Rotation.from_euler("ZXZ", np.column_stack((np.pi, np.pi, 0))).inv()
    RZ = Rotation.from_euler("ZXZ", np.column_stack((np.pi, 0, 0))).inv()
    transform_matrices = [I, RX, RY, RZ]
    transform_matrix_names = ["I", "Rx", "Ry", "Rz"]
    # Compute the symmetry-related sets of fitting parameters
    transform_labels, list_model_parameters = [], []
    for i in range(4):
        for j in range(4):
            new_model_parameters = deepcopy(model_parameters)
            for k in range(len(fitting_parameters["r_mean"])):
                transform_label = "{0}:{1}/{2}:{3}".format(
                    spin_labels[0], 
                    transform_matrix_names[i], 
                    spin_labels[1], 
                    transform_matrix_names[j]
                    )
                transform_labels.append(transform_label)
                # Set transformations
                transform_matrix1 = transform_matrices[i]
                transform_matrix2 = transform_matrices[j]
                # Set the initial values of angles
                xi_mean = new_model_parameters["xi_mean"][k]
                phi_mean =  new_model_parameters["phi_mean"][k]
                alpha_mean =  new_model_parameters["alpha_mean"][k]
                beta_mean = new_model_parameters["beta_mean"][k]
                gamma_mean = new_model_parameters["gamma_mean"][k]
                # Set the initial orientations of spins
                r_orientation = spherical2cartesian(1, xi_mean, phi_mean)
                spin_frame_rotation = Rotation.from_euler(
                    simulator.euler_angles_convention, 
                    np.column_stack((alpha_mean, beta_mean, gamma_mean))
                    ).inv()
                # Compute the symmetry-related orientations of spins
                new_r_orientation = transform_matrix1.apply(r_orientation)
                new_spin_frame_rotation = transform_matrix2 * spin_frame_rotation * transform_matrix1
                # Compute the symmetry-related values of angles
                spherical_coordinates = cartesian2spherical(new_r_orientation)
                new_rho_mean = spherical_coordinates[0][0] 
                new_xi_mean = spherical_coordinates[1][0] 
                new_phi_mean = spherical_coordinates[2][0]
                euler_angles = new_spin_frame_rotation.inv().as_euler(
                    simulator.euler_angles_convention, degrees = False
                    )
                new_alpha_mean = euler_angles[0][0]
                new_beta_mean = euler_angles[0][1]
                new_gamma_mean = euler_angles[0][2]
                # Store the symmetry-related values of angles
                new_model_parameters["xi_mean"][k] = new_xi_mean
                new_model_parameters["phi_mean"][k] = new_phi_mean
                new_model_parameters["alpha_mean"][k] = new_alpha_mean
                new_model_parameters["beta_mean"][k] = new_beta_mean
                new_model_parameters["gamma_mean"][k] = new_gamma_mean
            list_model_parameters.append(new_model_parameters)
    # Compute the score for the symmetry-related models
    run_with_mpi = get_mpi()
    if run_with_mpi:
        with MPIPoolExecutor() as executor:
            result = executor.map(scoring_function, list_model_parameters)
        score_values = list(result)
    else:
        pool = Pool()
        score_values = pool.map(scoring_function, list_model_parameters)
        pool.close()
        pool.join()
    # Store the results
    symmetry_related_models = []
    c = 0
    for i in range(4):
        for j in range(4):
            symmetry_related_model = {
                "transformation" : transform_labels[c],
                "parameters" : list_model_parameters[c],
                "score" : score_values[c],
                }
            symmetry_related_models.append(symmetry_related_model)
            c += 1
    return symmetry_related_models
    

def compute_model_after_spin_exchange(model_parameters, fitting_parameters, simulator):
    """Exchange spins A and B and calculates the values of xi, phi, alpha, beta, and gamma."""
    new_model_parameters = deepcopy(model_parameters)
    for k in range(len(fitting_parameters["r_mean"])):
        # Set the initial values of angles
        xi_mean = model_parameters["xi_mean"][k]
        xi_width = model_parameters["xi_width"][k]
        phi_mean = model_parameters["phi_mean"][k]
        phi_width = model_parameters["phi_width"][k]
        alpha_mean = model_parameters["alpha_mean"][k]
        alpha_width = model_parameters["alpha_width"][k]
        beta_mean = model_parameters["beta_mean"][k]
        beta_width = model_parameters["beta_width"][k]
        gamma_mean = model_parameters["gamma_mean"][k]
        gamma_width = model_parameters["gamma_width"][k]
        xi_values = random_samples_from_distribution(
            simulator.distribution_types["xi"], xi_mean, xi_width, 3 * simulator.num_samples, False
            )
        indices, = np.where(xi_values >= 0)
        pos_xi_values = xi_values[indices]
        xi_values = pos_xi_values[:simulator.num_samples]
        phi_values = random_samples_from_distribution(
            simulator.distribution_types["phi"], phi_mean, phi_width, simulator.num_samples, False
            )
        alpha_values = random_samples_from_distribution(
            simulator.distribution_types["alpha"], alpha_mean, alpha_width, simulator.num_samples, False
            )
        beta_values = random_samples_from_distribution(
            simulator.distribution_types["beta"], beta_mean, beta_width, 3 * simulator.num_samples, False
            )
        indices, = np.where(beta_values >= 0)
        pos_beta_values = beta_values[indices]
        beta_values = pos_beta_values[:simulator.num_samples]
        gamma_values = random_samples_from_distribution(
            simulator.distribution_types["gamma"], gamma_mean, gamma_width, simulator.num_samples, False
            )
        # Plot the distributions of parameters
        # from plots.monte_carlo.plot_monte_carlo_points import plot_monte_carlo_points
        # plot_monte_carlo_points(
            # [], xi_values, phi_values, alpha_values, beta_values, gamma_values, [], 
            # "parameter_distributions_initial.png"
            # )
        # Set the initial orientations of spins
        r_orientations = spherical2cartesian(np.ones(simulator.num_samples), xi_values, phi_values)
        spin_frame_rotations = Rotation.from_euler(
            simulator.euler_angles_convention, 
            np.column_stack((alpha_values, beta_values, gamma_values))
            ).inv()
        # Compute the orientations of spins in the coordinate frame of spin B
        new_r_orientations = spin_frame_rotations.apply(-1 * r_orientations)
        spherical_coordinates = cartesian2spherical(new_r_orientations)
        new_rho_values = spherical_coordinates[0]
        new_xi_values = spherical_coordinates[1]
        new_phi_values = spherical_coordinates[2]
        new_spin_frame_rotations = spin_frame_rotations.inv()
        euler_angles = new_spin_frame_rotations.inv().as_euler(
            simulator.euler_angles_convention, degrees = False
            )
        new_alpha_values = euler_angles[:,0]
        new_beta_values = euler_angles[:,1]
        new_gamma_values = euler_angles[:,2]        
        # Plot the distributions of parameters
        # from plots.monte_carlo.plot_monte_carlo_points import plot_monte_carlo_points
        # plot_monte_carlo_points(
            # [], new_xi_values, new_phi_values, new_alpha_values, new_beta_values, new_gamma_values, [], 
            # "parameter_distributions.png"
            # )
        # Compute the distributions
        xi_grid, xi_probs = compute_distribution(new_xi_values, 0, np.pi, np.pi / 180)
        phi_grid, phi_probs = compute_distribution(new_phi_values, -np.pi, np.pi, np.pi / 180)
        alpha_grid, alpha_probs = compute_distribution(new_alpha_values, -np.pi, np.pi, np.pi / 180)
        beta_grid, beta_probs = compute_distribution(new_beta_values, 0, np.pi, np.pi / 180)
        gamma_grid, gamma_probs = compute_distribution(new_gamma_values, -np.pi, np.pi, np.pi / 180)
        # Fit the distributions and determine the mean values and the widths
        new_xi_mean, new_xi_width = fit_distribution(
            xi_grid, xi_probs, simulator.distribution_types["xi"]
            )
        new_phi_mean, new_phi_width = fit_distribution(
            phi_grid, phi_probs, simulator.distribution_types["phi"]
            )
        new_alpha_mean, new_alpha_width = fit_distribution(
            alpha_grid, alpha_probs, simulator.distribution_types["alpha"]
            )
        new_beta_mean, new_beta_width = fit_distribution(
            beta_grid, beta_probs, simulator.distribution_types["beta"]
            )
        new_gamma_mean, new_gamma_width = fit_distribution(
            gamma_grid, gamma_probs, simulator.distribution_types["gamma"]
            )
        # Store parameters
        new_model_parameters["xi_mean"][k] = new_xi_mean
        new_model_parameters["xi_width"][k] = new_xi_width
        new_model_parameters["phi_mean"][k] = new_phi_mean
        new_model_parameters["phi_width"][k] = new_phi_width
        new_model_parameters["alpha_mean"][k] = new_alpha_mean
        new_model_parameters["alpha_width"][k] = new_alpha_width
        new_model_parameters["beta_mean"][k] = new_beta_mean
        new_model_parameters["beta_width"][k] = new_beta_width
        new_model_parameters["gamma_mean"][k] = new_gamma_mean
        new_model_parameters["gamma_width"][k] = new_gamma_width
    return new_model_parameters
    

def compute_distribution(x, x_min, x_max, x_inc):
    """Compute the distribution of a parameter."""
    x_grid = np.arange(x_min, x_max + x_inc, x_inc)
    probs = histogram(x, bins = x_grid)
    probs = probs / np.amax(probs)
    return x_grid, probs


def fit_distribution(x, prob, distribution_type):
    """Fit a distribution to a uniform distribution, a Gaussian distribution or 
    a von Mises distribution."""
    indices, = np.where(prob > 0)
    if indices.size == 1:
        mean, width = x[indices[0]], 0.0
        return mean, width
    else:
        if distribution_type == "uniform":
            popt, pcov = curve_fit(
                uniform_distribution, x[1:], prob[1:], maxfev = 1000000
                )
            mean, width = popt[1], np.abs(popt[2])
        elif distribution_type == "normal":
            popt, pcov = curve_fit(
                normal_distribution, x[1:], prob[1:], maxfev = 1000000
                )
            mean, width = popt[1], np.abs(popt[2])
        elif distribution_type == "vonmises":
            popt, pcov = curve_fit(
                vonmises_distribution, x[1:], prob[1:], maxfev = 1000000
                )
            mean, width = popt[1], np.abs(popt[2])
        return mean, width


def uniform_distribution(x, a, mean, width):
    """Uniform distribution."""
    return a * np.where((x >= mean - 0.5 * width) & (x <= mean + 0.5 * width), 1.0, 0.0)


def normal_distribution(x, a, mean, width):
    """Gaussian distribution."""
    std = width * const["fwhm2std"]
    if std == 0:
        increment = x[1] - x[0]
        return a * np.where(x - mean < increment, 1.0, 0.0)
    else:
        return a * np.exp(-0.5 * ((x - mean) / std)**2) / (np.sqrt(2 * np.pi) * std)


def vonmises_distribution(x, a, mean, width):
    """Von Mises distribution."""
    std = width * const["fwhm2std"]
    if std == 0:
       increment = x[1] - x[0]
       return a * np.where(x - mean < increment, 1.0, 0.0)
    else:
        kappa =  1 / std**2
        if np.isfinite(i0(kappa)):
            return a * np.exp(kappa * np.cos(x - mean)) / (2 * np.pi * i0(kappa))
        else:
            return a * np.exp(-0.5 * ((x - mean) / std)**2) / (np.sqrt(2 * np.pi) * std)