import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation 
sys.path.append(os.getcwd())
from mathematics.rotation_matrix import RotationMatrix

''' Comparison of the rotation matrix created by scipy and the rotation matrix from
    rotation_matrix.py '''

deg_to_rad = np.pi/180
alpha = 90 * deg_to_rad
beta = 30 * deg_to_rad
gamma = 45 * deg_to_rad

rot_matrix_from_script = RotationMatrix()
rot_matrix_from_script.reset_angles(alpha, beta, gamma)
print(rot_matrix_from_script.rotation_matrix)


rot_matrix_scipy = Rotation.from_euler('ZXZ', [alpha, beta, gamma], degrees = False)
print(rot_matrix_scipy.as_matrix())

""" The scipy method and the python script produce the same rotation matrix. 
    To avoid unnecessary code, scipy will be used and rotation_matrix.py will be deleted """


vectors = np.array(((0.1, 0.2 , 0.3), (0.4, 0.5, 0.6)))
rotated_vectors = rot_matrix_scipy.apply(vectors)
print(rotated_vectors)
rotated_vectors = rot_matrix_scipy.as_matrix().dot(vectors)



