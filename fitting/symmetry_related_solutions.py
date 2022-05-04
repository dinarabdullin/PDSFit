import sys
import numpy as np
from copy import deepcopy
import itertools
from multiprocessing import Pool
from scipy.spatial.transform import Rotation
from mathematics.coordinate_system_conversions import spherical2cartesian, cartesian2spherical
from mathematics.rotate_coordinate_system import rotate_coordinate_system
from fitting.check_relative_weights import check_relative_weights


def merge_fitted_and_fixed_variables(variables_indices, fitted_variables_values, fixed_variables_values):
    ''' Merges fitted and fixed variables into a single dictionary '''
    # Check/correct the relative weights
    new_fitted_variables_values = check_relative_weights(variables_indices, fitted_variables_values, fixed_variables_values)
    # Merge variables
    all_variables = {}
    for variable_name in variables_indices:
        variable_indices = variables_indices[variable_name]
        list_variable_values = []
        for i in range(len(variable_indices)):
            sublist_variable_values = []
            for j in range(len(variable_indices[i])):
                variable_object = variable_indices[i][j]
                if variable_object.optimize:
                    variable_value = new_fitted_variables_values[variable_object.index]
                else:
                    variable_value = fixed_variables_values[variable_object.index]
                sublist_variable_values.append(variable_value)
            list_variable_values.append(sublist_variable_values)
        all_variables[variable_name] = list_variable_values
    return all_variables

    
def compute_symmetry_related_solutions(variables_indices, fitted_variables_values, fixed_variables_values, simulator, score_function):
    ''' Computes symmetry-related sets of fitting parameters '''
    # Merge fitted variables and fixed variables into a single dictionary
    all_variables = merge_fitted_and_fixed_variables(variables_indices, fitted_variables_values, fixed_variables_values)
    # Transformation matrices
    I = Rotation.from_euler('ZXZ', np.column_stack((0, 0, 0)))
    RX = Rotation.from_euler('ZXZ', np.column_stack((0, np.pi, 0))).inv()
    RY = Rotation.from_euler('ZXZ', np.column_stack((np.pi, np.pi, 0))).inv()
    RZ = Rotation.from_euler('ZXZ', np.column_stack((np.pi, 0, 0))).inv()
    transformation_matrices = [I, RX, RY, RZ]
    transformation_matrix_names = ['I', 'Rx', 'Ry', 'Rz']
    # Transformation matrices for entire spin system
    n_spins = len(variables_indices['r_mean']) + 1 
    transformations = list(itertools.product(transformation_matrices, repeat=n_spins))
    transformation_names = list(itertools.product(transformation_matrix_names, repeat=n_spins))
    #print(transformation_names)
    # Compute the symmetry-related sets of fitting parameters
    symmetry_related_solutions = []
    for count, transformation in enumerate(transformations):
        new_variables = deepcopy(all_variables)
        for i in range(n_spins-1):
            n_components = len(variables_indices['r_mean'][i])
            for j in range(n_components):
                # Set the transormations
                transformation_matrix1 = transformation[0]
                transformation_matrix2 = transformation[i+1]
                # Get the initial values of angles
                xi = new_variables['xi_mean'][i][j]
                phi =  new_variables['phi_mean'][i][j]
                alpha =  new_variables['alpha_mean'][i][j]
                beta = new_variables['beta_mean'][i][j]
                gamma = new_variables['gamma_mean'][i][j]
                # Get the initial orientations of spins
                r_orientation = spherical2cartesian(1, xi, phi)
                spin_frame_rotation = Rotation.from_euler(simulator.euler_angles_convention, np.column_stack((alpha, beta, gamma))).inv()
                # Compute the symmetry-related orientations of spins
                new_r_orientation = rotate_coordinate_system(r_orientation, transformation_matrix1, simulator.separate_grids)
                new_spin_frame_rotation = transformation_matrix2 * spin_frame_rotation * transformation_matrix1
                # Compute the symmetry-related values of angles
                #new_rho, new_xi, new_phi = cartesian2spherical(new_r_orientation)
                spherical_coordinates = cartesian2spherical(new_r_orientation)
                new_rho, new_xi, new_phi = spherical_coordinates[1][0], spherical_coordinates[1][0], spherical_coordinates[2][0]
                euler_angles = new_spin_frame_rotation.inv().as_euler(simulator.euler_angles_convention, degrees=False)
                new_alpha, new_beta, new_gamma = euler_angles[0,0], euler_angles[0,1], euler_angles[0,2]
                # Format the symmetry-related values of angles
                new_xi = np.where(new_xi < 0, -new_xi, new_xi)
                new_phi = np.where(new_phi < 0, new_phi + 2*np.pi, new_phi)
                new_alpha = np.where(new_alpha < 0, new_alpha + 2*np.pi, new_alpha)
                new_beta = np.where(new_beta < 0, -new_beta, new_beta)
                new_gamma = np.where(new_gamma < 0, new_gamma + 2*np.pi, new_gamma)
                # Store the symmetry-related values of angles
                new_variables['xi_mean'][i][j] = new_xi
                new_variables['phi_mean'][i][j] = new_phi
                new_variables['alpha_mean'][i][j] = new_alpha
                new_variables['beta_mean'][i][j] = new_beta
                new_variables['gamma_mean'][i][j] = new_gamma
        # Compute the score for the symmetry-related sets of fitting parameters
        score = score_function(new_variables)
        # Store the results
        symmetry_related_solution = {
            'transformation' : ''.join(transformation_names[count]),
            'variables'      : new_variables,
            'score'          : score,
        }
        symmetry_related_solutions.append(symmetry_related_solution)  
    return symmetry_related_solutions