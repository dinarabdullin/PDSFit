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
#print(rot_matrix_from_script.rotation_matrix)


rot_matrix_scipy = Rotation.from_euler('ZXZ', [alpha, beta, gamma])
#print(rot_matrix_scipy.as_matrix())

""" The scipy method and the python script produce the same rotation matrix. 
    To avoid unnecessary code, scipy will be used and rotation_matrix.py will be deleted """

##############################
angles = np.array([[90, 0, 0],[0, 45, 0],[45, 60, 30]])
vectors = np.random.normal(0, 1 , size=(3, 3))
rot_matrix_scipy = Rotation.from_euler('ZXZ', angles, degrees=True)
rotated_vectors = rot_matrix_scipy.apply(vectors)
print(rotated_vectors)