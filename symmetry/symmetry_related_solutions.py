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
from mpi.mpi_status import get_mpi
from mathematics.histogram import histogram
from mathematics.coordinate_system_conversions import spherical2cartesian, cartesian2spherical
from mathematics.find_nearest import find_nearest
from mathematics.random_points_from_distribution import random_points_from_distribution
from fitting.merge_parameters import merge_parameters
from supplement.definitions import const
     

def compute_symmetry_related_solutions(optimized_model_parameters, fitting_parameters, simulator, score_function, spins):
    ''' Computes symmetry-related sets of fitting parameters '''
    sys.stdout.write('\nComputing the symmetry-related sets of model parameters... ')
    sys.stdout.flush() 
    all_symmetry_related_solutions = []
    # Merge the optimized and fixed model parameters into a single dictionary
    model_parameters = merge_parameters(optimized_model_parameters, fitting_parameters)
    if spins[0] == spins[1]:
        # For the assignement: spin 1 = spin A, spin 2 = spin B
        spin_labels = ['A','B']
        symmetry_related_solutions_set1 = compute_equvivalent_angles(model_parameters, fitting_parameters, simulator, score_function, spin_labels)
        # For the assignement: spin 1 = spin B, spin 2 = spin A
        spin_labels = ['B','A']
        model_parameters_set2 = compute_model_parameters_after_spin_exchange(model_parameters, fitting_parameters, simulator)
        symmetry_related_solutions_set2 = compute_equvivalent_angles(model_parameters_set2, fitting_parameters, simulator, score_function, spin_labels)
        symmetry_related_solutions = symmetry_related_solutions_set1 + symmetry_related_solutions_set2
    else:
        spin_labels = ['A','B']
        symmetry_related_solutions = compute_equvivalent_angles(model_parameters, fitting_parameters, simulator, score_function, spin_labels)
    sys.stdout.write('done!\n')
    sys.stdout.flush() 
    return symmetry_related_solutions


def compute_equvivalent_angles(model_parameters, fitting_parameters, simulator, score_function, spin_labels):
    ''' Computes 16 sets of equvivalent angles '''
    # Transformation matrices
    I = Rotation.from_euler('ZXZ', np.column_stack((0, 0, 0)))
    RX = Rotation.from_euler('ZXZ', np.column_stack((0, np.pi, 0))).inv()
    RY = Rotation.from_euler('ZXZ', np.column_stack((np.pi, np.pi, 0))).inv()
    RZ = Rotation.from_euler('ZXZ', np.column_stack((np.pi, 0, 0))).inv()
    transformation_matrices = [I, RX, RY, RZ]
    transformation_matrix_names = ['I', 'Rx', 'Ry', 'Rz']
    # Compute the symmetry-related sets of fitting parameters
    model_parameters_all_sets = []
    for i in range(4):
        for j in range(4):
            new_model_parameters = deepcopy(model_parameters)
            n_components = len(fitting_parameters['indices']['r_mean'])
            for k in range(n_components):
                # Set the transformations
                transformation_matrix1 = transformation_matrices[i]
                transformation_matrix2 = transformation_matrices[j]
                # Set the initial values of angles
                xi_mean = new_model_parameters['xi_mean'][k]
                phi_mean =  new_model_parameters['phi_mean'][k]
                alpha_mean =  new_model_parameters['alpha_mean'][k]
                beta_mean = new_model_parameters['beta_mean'][k]
                gamma_mean = new_model_parameters['gamma_mean'][k]
                # Set the initial orientations of spins
                r_orientation = spherical2cartesian(1, xi_mean, phi_mean)
                spin_frame_rotation = Rotation.from_euler(simulator.euler_angles_convention, np.column_stack((alpha_mean, beta_mean, gamma_mean))).inv()
                # Compute the symmetry-related orientations of spins
                new_r_orientation = transformation_matrix1.apply(r_orientation)
                new_spin_frame_rotation = transformation_matrix2 * spin_frame_rotation * transformation_matrix1
                # Compute the symmetry-related values of angles
                spherical_coordinates = cartesian2spherical(new_r_orientation)
                new_rho_mean, new_xi_mean, new_phi_mean = spherical_coordinates[0][0], spherical_coordinates[1][0], spherical_coordinates[2][0]
                euler_angles = new_spin_frame_rotation.inv().as_euler(simulator.euler_angles_convention, degrees=False)
                new_alpha_mean, new_beta_mean, new_gamma_mean = euler_angles[0][0], euler_angles[0][1], euler_angles[0][2]
                # Store the symmetry-related values of angles
                new_model_parameters['xi_mean'][k] = new_xi_mean
                new_model_parameters['phi_mean'][k] = new_phi_mean
                new_model_parameters['alpha_mean'][k] = new_alpha_mean
                new_model_parameters['beta_mean'][k] = new_beta_mean
                new_model_parameters['gamma_mean'][k] = new_gamma_mean
                model_parameters_all_sets.append(new_model_parameters)
                # # Set the transformations
                # transformation_matrix1 = transformation_matrices[i]
                # transformation_matrix2 = transformation_matrices[j]
                # # Set the initial values of angles
                # xi_mean = [model_parameters['xi_mean'][k]]
                # xi_width = [model_parameters['xi_width'][k]]
                # phi_mean = [model_parameters['phi_mean'][k]]
                # phi_width = [model_parameters['phi_width'][k]]
                # alpha_mean = [model_parameters['alpha_mean'][k]]
                # alpha_width = [model_parameters['alpha_width'][k]]
                # beta_mean = [model_parameters['beta_mean'][k]]
                # beta_width = [model_parameters['beta_width'][k]]
                # gamma_mean = [model_parameters['gamma_mean'][k]]
                # gamma_width = [model_parameters['gamma_width'][k]] 
                # rel_prob = [1.0]
                # xi_values = random_points_from_distribution(simulator.distribution_types['xi'], xi_mean, xi_width, rel_prob, 3*simulator.num_samples, False)
                # idx, = np.where(xi_values >= 0)
                # pos_xi_values = xi_values[idx]
                # xi_values = pos_xi_values[0:simulator.num_samples]
                # phi_values = random_points_from_distribution(simulator.distribution_types['phi'], phi_mean, phi_width, rel_prob, simulator.num_samples, False)
                # alpha_values = random_points_from_distribution(simulator.distribution_types['alpha'], alpha_mean, alpha_width, rel_prob, simulator.num_samples, False)
                # beta_values = random_points_from_distribution(simulator.distribution_types['beta'], beta_mean, beta_width, rel_prob, 3*simulator.num_samples, False)
                # idx, = np.where(beta_values >= 0)
                # pos_beta_values = beta_values[idx]
                # beta_values = pos_beta_values[0:simulator.num_samples]
                # gamma_values = random_points_from_distribution(simulator.distribution_types['gamma'], gamma_mean, gamma_width, rel_prob, simulator.num_samples, False)
                # # Plot the distributions of parameters
                # from plots.monte_carlo.plot_monte_carlo_points import plot_monte_carlo_points
                # plot_monte_carlo_points([], xi_values, phi_values, alpha_values, beta_values, gamma_values, [], 'parameter_distributions_initial.png')
                # # Set the initial orientations of spins
                # r_orientations = spherical2cartesian(np.ones(simulator.num_samples), xi_values, phi_values)
                # spin_frame_rotations = Rotation.from_euler(simulator.euler_angles_convention, np.column_stack((alpha_values, beta_values, gamma_values))).inv()
                # # Compute the symmetry-related values of angles
                # new_r_orientations = transformation_matrix1.apply(r_orientations)
                # spherical_coordinates = cartesian2spherical(new_r_orientations)
                # new_rho_values, new_xi_values, new_phi_values = spherical_coordinates[0], spherical_coordinates[1], spherical_coordinates[2]
                # new_spin_frame_rotations = transformation_matrix2 * spin_frame_rotations * transformation_matrix1
                # euler_angles = new_spin_frame_rotations.inv().as_euler(simulator.euler_angles_convention, degrees=False)
                # new_alpha_values, new_beta_values, new_gamma_values = euler_angles[:,0], euler_angles[:,1], euler_angles[:,2] 
                # # Plot the distributions of parameters
                # from plots.monte_carlo.plot_monte_carlo_points import plot_monte_carlo_points
                # plot_monte_carlo_points([], new_xi_values, new_phi_values, new_alpha_values, new_beta_values, new_gamma_values, [], 'parameter_distributions_transform.png')
                # # Set the fixed parameters
                # xi_mean_object = fitting_parameters['indices']['xi_mean'][k]
                # xi_width_object = fitting_parameters['indices']['xi_width'][k]
                # phi_mean_object = fitting_parameters['indices']['phi_mean'][k]
                # phi_width_object = fitting_parameters['indices']['phi_width'][k]
                # alpha_mean_object = fitting_parameters['indices']['alpha_mean'][k]
                # alpha_width_object = fitting_parameters['indices']['alpha_width'][k]
                # beta_mean_object = fitting_parameters['indices']['beta_mean'][k]
                # beta_width_object = fitting_parameters['indices']['beta_width'][k]
                # gamma_mean_object = fitting_parameters['indices']['gamma_mean'][k]
                # gamma_width_object = fitting_parameters['indices']['gamma_width'][k]
                # xi_fixed, phi_fixed = 0, 0
                # if not xi_mean_object.optimize and not xi_width_object.optimize and fitting_parameters['values'][xi_width_object.index] == 0:
                    # xi_fixed = fitting_parameters['values'][xi_mean_object.index]
                    # xi_corr = new_xi_values - xi_fixed
                # else:
                    # xi_corr = np.zeros(xi_values.size)
                # if not phi_mean_object.optimize and not phi_width_object.optimize and fitting_parameters['values'][phi_width_object.index] == 0:
                    # phi_fixed = fitting_parameters['values'][phi_mean_object.index]
                    # phi_corr = new_phi_values - phi_fixed
                # else:
                    # phi_corr = np.zeros(phi_values.size)
                # correction_matrix1 = Rotation.from_euler('ZYZ', np.column_stack((phi_corr, xi_corr, np.zeros(phi_corr.size)))).inv()
                # new_r_orientations = correction_matrix1.apply(new_r_orientations)
                # spherical_coordinates = cartesian2spherical(new_r_orientations)
                # new_rho_values, new_xi_values, new_phi_values = spherical_coordinates[0], spherical_coordinates[1], spherical_coordinates[2]
                # new_spin_frame_rotations = new_spin_frame_rotations * correction_matrix1
                # euler_angles = new_spin_frame_rotations.inv().as_euler(simulator.euler_angles_convention, degrees=False)
                # new_alpha_values, new_beta_values, new_gamma_values = euler_angles[:,0], euler_angles[:,1], euler_angles[:,2]
                # alpha_fixed, beta_fixed, gamma_fixed = 0, 0, 0
                # if not alpha_mean_object.optimize and not alpha_width_object.optimize and fitting_parameters['values'][alpha_width_object.index] == 0:
                    # alpha_fixed = fitting_parameters['values'][alpha_mean_object.index]
                    # alpha_corr = alpha_fixed - new_alpha_values
                # else:
                    # alpha_corr = np.zeros(alpha_values.size)
                # if not beta_mean_object.optimize and not beta_width_object.optimize and fitting_parameters['values'][beta_width_object.index] == 0:
                    # beta_fixed = fitting_parameters['values'][beta_mean_object.index]
                    # beta_corr = beta_fixed - new_beta_values
                # else:
                    # beta_corr = np.zeros(beta_values.size)
                # if not gamma_mean_object.optimize and not gamma_width_object.optimize and fitting_parameters['values'][gamma_width_object.index] == 0:
                    # gamma_fixed = fitting_parameters['values'][gamma_mean_object.index]
                    # gamma_corr = gamma_fixed - new_gamma_values
                # else:
                    # gamma_corr = np.zeros(gamma_values.size)
                # correction_matrix2 = Rotation.from_euler(simulator.euler_angles_convention, np.column_stack((alpha_corr, beta_corr, gamma_corr))).inv()
                # new_spin_frame_rotations = correction_matrix2 * new_spin_frame_rotations
                # euler_angles = new_spin_frame_rotations.inv().as_euler(simulator.euler_angles_convention, degrees=False)
                # new_alpha_values, new_beta_values, new_gamma_values = euler_angles[:,0], euler_angles[:,1], euler_angles[:,2]
                # # Plot the distributions of parameters
                # from plots.monte_carlo.plot_monte_carlo_points import plot_monte_carlo_points
                # plot_monte_carlo_points([], new_xi_values, new_phi_values, new_alpha_values, new_beta_values, new_gamma_values, [], 'parameter_distributions_corr.png') 
                # # Compute the distributions
                # xi_grid, xi_probs = compute_distribution(new_xi_values, 0, np.pi, np.pi/1800)
                # phi_grid, phi_probs = compute_distribution(new_phi_values, -np.pi, np.pi, np.pi/1800)
                # alpha_grid, alpha_probs = compute_distribution(new_alpha_values, -np.pi, np.pi, np.pi/1800)
                # beta_grid, beta_probs = compute_distribution(new_beta_values, 0, np.pi, np.pi/1800)
                # gamma_grid, gamma_probs = compute_distribution(new_gamma_values, -np.pi, np.pi, np.pi/1800)
                # # Fit the distributions and determine the mean values and the widths
                # new_xi_mean, new_xi_width = fit_distribution(xi_grid, xi_probs, simulator.distribution_types['xi'])
                # new_phi_mean, new_phi_width = fit_distribution(phi_grid, phi_probs, simulator.distribution_types['phi'])
                # new_alpha_mean, new_alpha_width = fit_distribution(alpha_grid, alpha_probs, simulator.distribution_types['alpha'])
                # new_beta_mean, new_beta_width = fit_distribution(beta_grid, beta_probs, simulator.distribution_types['beta'])
                # new_gamma_mean, new_gamma_width = fit_distribution(gamma_grid, gamma_probs, simulator.distribution_types['gamma'])
                # # Store the symmetry-related values of angles
                # new_model_parameters['xi_mean'][k] = new_xi_mean
                # new_model_parameters['xi_width'][k] = new_xi_width
                # new_model_parameters['phi_mean'][k] = new_phi_mean
                # new_model_parameters['phi_width'][k] = new_phi_width
                # new_model_parameters['alpha_mean'][k] = new_alpha_mean
                # new_model_parameters['alpha_width'][k] = new_alpha_width
                # new_model_parameters['beta_mean'][k] = new_beta_mean
                # new_model_parameters['beta_width'][k] = new_beta_width
                # new_model_parameters['gamma_mean'][k] = new_gamma_mean
                # new_model_parameters['gamma_width'][k] = new_gamma_width
                # model_parameters_all_sets.append(new_model_parameters)
    # Compute the score for the symmetry-related sets of fitting parameters
    run_with_mpi = get_mpi()
    if run_with_mpi:
        with MPIPoolExecutor() as executor:
            result = executor.map(score_function, model_parameters_all_sets)
        score_values = list(result)
    else:
        pool = Pool()
        score_values = pool.map(score_function, model_parameters_all_sets)
        pool.close()
        pool.join()
    # Store the results
    symmetry_related_solutions = []
    c = 0
    for i in range(4):
        for j in range(4):
            symmetry_related_solution = {
                'transformation' : '{0}:{1}/{2}:{3}'.format(spin_labels[0], transformation_matrix_names[i], spin_labels[1], transformation_matrix_names[j]),
                'variables'      : model_parameters_all_sets[c],
                'score'          : score_values[c],
            }
            symmetry_related_solutions.append(symmetry_related_solution)
            c += 1
    return symmetry_related_solutions
    

def compute_model_parameters_after_spin_exchange(model_parameters, fitting_parameters, simulator):
    ''' Exchanges spins A and B and calculates the values of xi, phi, alpha, beta, and gamma angles '''
    new_model_parameters = deepcopy(model_parameters)
    n_components = len(fitting_parameters['indices']['r_mean'])
    for k in range(n_components):
        # Set the initial values of angles
        xi_mean = [model_parameters['xi_mean'][k]]
        xi_width = [model_parameters['xi_width'][k]]
        phi_mean = [model_parameters['phi_mean'][k]]
        phi_width = [model_parameters['phi_width'][k]]
        alpha_mean = [model_parameters['alpha_mean'][k]]
        alpha_width = [model_parameters['alpha_width'][k]]
        beta_mean = [model_parameters['beta_mean'][k]]
        beta_width = [model_parameters['beta_width'][k]]
        gamma_mean = [model_parameters['gamma_mean'][k]]
        gamma_width = [model_parameters['gamma_width'][k]] 
        rel_prob = [1.0]
        xi_values = random_points_from_distribution(simulator.distribution_types['xi'], xi_mean, xi_width, rel_prob, 3*simulator.num_samples, False)
        idx, = np.where(xi_values >= 0)
        pos_xi_values = xi_values[idx]
        xi_values = pos_xi_values[0:simulator.num_samples]
        phi_values = random_points_from_distribution(simulator.distribution_types['phi'], phi_mean, phi_width, rel_prob, simulator.num_samples, False)
        alpha_values = random_points_from_distribution(simulator.distribution_types['alpha'], alpha_mean, alpha_width, rel_prob, simulator.num_samples, False)
        beta_values = random_points_from_distribution(simulator.distribution_types['beta'], beta_mean, beta_width, rel_prob, 3*simulator.num_samples, False)
        idx, = np.where(beta_values >= 0)
        pos_beta_values = beta_values[idx]
        beta_values = pos_beta_values[0:simulator.num_samples]
        gamma_values = random_points_from_distribution(simulator.distribution_types['gamma'], gamma_mean, gamma_width, rel_prob, simulator.num_samples, False)
        # # Plot the distributions of parameters
        # from plots.monte_carlo.plot_monte_carlo_points import plot_monte_carlo_points
        # plot_monte_carlo_points([], xi_values, phi_values, alpha_values, beta_values, gamma_values, [], 'parameter_distributions_initial.png')
        # Set the initial orientations of spins
        r_orientations = spherical2cartesian(np.ones(simulator.num_samples), xi_values, phi_values)
        spin_frame_rotations = Rotation.from_euler(simulator.euler_angles_convention, np.column_stack((alpha_values, beta_values, gamma_values))).inv()
        # Compute the orientations of spins in the coordinate frame of spin B
        new_r_orientations = spin_frame_rotations.apply(-1 * r_orientations)
        spherical_coordinates = cartesian2spherical(new_r_orientations)
        new_rho_values, new_xi_values, new_phi_values = spherical_coordinates[0], spherical_coordinates[1], spherical_coordinates[2]
        new_spin_frame_rotations = spin_frame_rotations.inv()
        euler_angles = new_spin_frame_rotations.inv().as_euler(simulator.euler_angles_convention, degrees=False)
        new_alpha_values, new_beta_values, new_gamma_values = euler_angles[:,0], euler_angles[:,1], euler_angles[:,2]        
        # # Plot the distributions of parameters
        # from plots.monte_carlo.plot_monte_carlo_points import plot_monte_carlo_points
        # plot_monte_carlo_points([], new_xi_values, new_phi_values, new_alpha_values, new_beta_values, new_gamma_values, [], 'parameter_distributions.png')
        # Compute the distributions
        xi_grid, xi_probs = compute_distribution(new_xi_values, 0, np.pi, np.pi/1800)
        phi_grid, phi_probs = compute_distribution(new_phi_values, -np.pi, np.pi, np.pi/1800)
        alpha_grid, alpha_probs = compute_distribution(new_alpha_values, -np.pi, np.pi, np.pi/1800)
        beta_grid, beta_probs = compute_distribution(new_beta_values, 0, np.pi, np.pi/1800)
        gamma_grid, gamma_probs = compute_distribution(new_gamma_values, -np.pi, np.pi, np.pi/1800)
        # Fit the distributions and determine the mean values and the widths
        new_xi_mean, new_xi_width = fit_distribution(xi_grid, xi_probs, simulator.distribution_types['xi'])
        new_phi_mean, new_phi_width = fit_distribution(phi_grid, phi_probs, simulator.distribution_types['phi'])
        new_alpha_mean, new_alpha_width = fit_distribution(alpha_grid, alpha_probs, simulator.distribution_types['alpha'])
        new_beta_mean, new_beta_width = fit_distribution(beta_grid, beta_probs, simulator.distribution_types['beta'])
        new_gamma_mean, new_gamma_width = fit_distribution(gamma_grid, gamma_probs, simulator.distribution_types['gamma'])
        # Store parameters
        new_model_parameters['xi_mean'][k] = new_xi_mean
        new_model_parameters['xi_width'][k] = new_xi_width
        new_model_parameters['phi_mean'][k] = new_phi_mean
        new_model_parameters['phi_width'][k] = new_phi_width
        new_model_parameters['alpha_mean'][k] = new_alpha_mean
        new_model_parameters['alpha_width'][k] = new_alpha_width
        new_model_parameters['beta_mean'][k] = new_beta_mean
        new_model_parameters['beta_width'][k] = new_beta_width
        new_model_parameters['gamma_mean'][k] = new_gamma_mean
        new_model_parameters['gamma_width'][k] = new_gamma_width
        # new_model_parameters['alpha_mean'][k] = -1 * model_parameters['gamma_mean'][k]
        # new_model_parameters['alpha_width'][k] = model_parameters['gamma_width'][k]
        # new_model_parameters['beta_mean'][k] = -1 * model_parameters['beta_mean'][k]
        # new_model_parameters['beta_width'][k] = model_parameters['beta_width'][k]
        # new_model_parameters['gamma_mean'][k] = -1 * model_parameters['alpha_mean'][k]
        # new_model_parameters['gamma_width'][k] = model_parameters['alpha_width'][k]
    return new_model_parameters
    

def compute_distribution(points, minimum, maximum, increment):
    values = np.arange(minimum, maximum+increment, increment)
    probabilities = histogram(points, bins=values)
    probabilities = probabilities / np.amax(probabilities)
    return values, probabilities


def fit_distribution(x, p, distribution_type):
    idx, = np.where(p > 0)
    if idx.size == 1:
        mean, width = x[idx[0]], 0.0
        return mean, width
    else:
        if distribution_type == "uniform":
            popt, pcov = curve_fit(uniform_distribution, x[1:], p[1:], maxfev=10000)
            mean, width = popt[1], np.abs(popt[2])
        elif distribution_type == "normal":
            popt, pcov = curve_fit(normal_distribution, x[1:], p[1:], maxfev=10000)
            mean, width = popt[1], np.abs(popt[2])
        elif distribution_type == "vonmises":
            popt, pcov = curve_fit(vonmises_distribution, x[1:], p[1:], maxfev=10000)
            mean, width = popt[1], np.abs(popt[2])
        else:
            raise ValueError('Unsupported didtribution type!')
            sys.exit(1)
        return mean, width


def uniform_distribution(x, A, mean, width):
    return A * np.where((x >= mean-0.5*width) & (x <= mean+0.5*width), 1.0, 0.0)


def normal_distribution(x, A, mean, width):
    std = width * const['fwhm2std']
    if std == 0:
        increment = x[1] - x[0]
        return A * np.where(x - mean < increment, 1.0, 0.0)
    else:
        return A * np.exp(-0.5 * ((x - mean)/std)**2) / (np.sqrt(2*np.pi) * std)


def vonmises_distribution(x, A, mean, width):
    std = width * const['fwhm2std']
    if std == 0:
       increment = x[1] - x[0]
       return A * np.where(x - mean < increment, 1.0, 0.0)
    else:
        kappa =  1 / std**2
        if np.isfinite(i0(kappa)):
            return A * np.exp(kappa * np.cos(x - mean)) / (2*np.pi * i0(kappa))
        else:
            return A * np.exp(-0.5 * ((x - mean)/std)**2) / (np.sqrt(2*np.pi) * std)