'''
The rotation matrix class
'''

# TODO: use numpy/scipy methods instead of this file
# TODO: ask Dinar what the transposed matrices/vectors are needed for

import numpy as np
from math import cos
from math import sin

class RotationMatrix:
    """
    3x3 Rotation matrix that follows the z-x-z convention of Euler Angles.
    In the z-x-z convention, the x-y-z frame is rotated three times:
    first about the z-axis by an angle alpha, then about the new x-axis by an
    angle beta, then about the newest z-axis by an angle gamma
    """
    
    rotation_matrix = np.zeros((3, 3))

    def __init__(self):
        """initialize the rotation matrix"""
        self.reset_angles(0, 0, 0)
        

    def reset_angles(self, alpha, beta, gamma):
        '''reset the rotation matrix to be able to perform the Euler 'z-x-z' active rotation 
        with the new angles alpha, beta, gamma'''
        c1 = cos(alpha)
        s1 = sin(alpha)
        c2 = cos(beta)
        s2 = sin(beta)
        c3 = cos(gamma)
        s3 = sin(gamma)
        self.rotation_matrix[0][0] = c1*c3 - c2*s1*s3
        self.rotation_matrix[0][1] = -c1*s3 - c2*c3*s1
        self.rotation_matrix[0][2] = s1*s2
        self.rotation_matrix[1][0] = c3*s1 + c1*c2*s3
        self.rotation_matrix[1][1] = c1*c2*c3 - s1*s3
        self.rotation_matrix[1][2] = -c1*s2
        self.rotation_matrix[2][0] = s2*s3
        self.rotation_matrix[2][1] = c3*s2
        self.rotation_matrix[2][2] = c2 
    
    def multiplication_with_vector(self, vector):
        """calculate the dot product of the rotation matrix and a vector"""
        new_vector = np.dot(self.rotation_matrix, vector)
        return new_vector

    def multiplication_with_matrix(self, matrix):
        """calculate the product of the rotation matrix and a 3x3 matrix"""
        new_matrix = np.dot(self.rotation_matrix, matrix)
        return new_matrix
