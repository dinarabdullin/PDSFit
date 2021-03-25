import sys
import numpy as np
from scipy.spatial.transform import Rotation
from mathematics.coordinate_system_conversions import spherical2cartesian, cartesian2spherical
from mathematics.rotate_coordinate_system import rotate_coordinate_system

    
def symmetry_related_orientations(self):
''' Computes symmetry-related orientations of spin system '''
    
    # C++ code
    # std::vector<std::vector<double>> sym_angles; sym_angles.reserve(16);
    # // The direction of the distance vector
    # std::vector<double> distDir; distDir.reserve(3);
    # distDir = direction_from_angles(xi, phi);
    # // Rotation matrices for the rotation between the spin frames
    # std::shared_ptr<RotationMatrix> RM(new RotationMatrix(alpha, beta, gamma));
    # // Rotation matrices for 180° rotation about the x,y,z axes of the coordinate system
    # std::shared_ptr<RotationMatrix> RI(new RotationMatrix(0.0, 0.0, 0.0));
    # std::shared_ptr<RotationMatrix> Rx(new RotationMatrix(0.0, PI, 0.0));
    # std::shared_ptr<RotationMatrix> Ry(new RotationMatrix(PI, PI, 0.0));
    # std::shared_ptr<RotationMatrix> Rz(new RotationMatrix(PI, 0.0, 0.0));
    # std::vector<std::shared_ptr<RotationMatrix>> transformations; transformations.reserve(4);
    # transformations.push_back(RI);
    # transformations.push_back(Rx);
    # transformations.push_back(Ry);
    # transformations.push_back(Rz);
    # // Calculate the symmetry-related sets of fitting parameters
    # std::vector<double> distDir_rotated; distDir_rotated.reserve(3);
    # std::vector<std::vector<double>> RM_rotated; RM_rotated.reserve(3);
    # std::vector<double> new_xi_phi; new_xi_phi.reserve(2);
    # std::vector<double> new_alpha_beta_gamma; new_alpha_beta_gamma.reserve(3);
    # std::vector<double> new_angles; new_angles.reserve(5);
    # for (size_t i = 0; i < 4; ++i) {
        # for (size_t j = 0; j < 4; ++j) {
            # // Rotate the spin A frame
            # distDir_rotated = transformations[i]->dot_product(distDir, true);
            # // Rotate the spin B frame
            # RM_rotated = transformations[i]->matrix_product(RM->matrix_product(transformations[j]->R, false), true);
            # // Calculate the new values of the angles
            # new_xi_phi = angles_from_direction(distDir_rotated);
            # new_alpha_beta_gamma = angles_from_rotation_matrix(RM_rotated);
            # // Save the calculated angles
            # new_angles.push_back(new_xi_phi[0]);
            # new_angles.push_back(new_xi_phi[1]);
            # new_angles.push_back(new_alpha_beta_gamma[0]);
            # new_angles.push_back(new_alpha_beta_gamma[1]);
            # new_angles.push_back(new_alpha_beta_gamma[2]);
            # sym_angles.push_back(new_angles);
            # new_angles.clear();
        # }
    # }
    # return sym_angles;