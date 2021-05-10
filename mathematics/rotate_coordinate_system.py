import numpy as np


def rotate_coordinate_system(vectors, rotation_matrices, separate_dimensions):
    ''' Rotate a reference coordinate frame using a rotation matrix '''
    if not separate_dimensions:
        return rotation_matrices.apply(vectors)
    else:
        L = vectors.shape[0]
        K = rotation_matrices.__len__()
        # Store rotated vectors in an array with a shape (L*K)x3
        for i in range(L):
            if i == 0:
                rotated_vectors = rotation_matrices.apply(vectors[i])
            else:
                rotated_vectors = np.vstack((rotated_vectors, rotation_matrices.apply(vectors[i])))
        # Store rotated vectors in an array with a shape NxMx3
        # rotated_vectors = np.zeros((N,M,3))
        # for i in range(N):
            # rotated_vectors[i,:,:] = rotation_matrices.apply(vectors[i])
        return rotated_vectors