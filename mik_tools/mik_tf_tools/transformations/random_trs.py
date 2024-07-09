import numpy as np
import torch
from scipy.stats import special_ortho_group
from mik_tools.mik_tf_tools.transformations.trs import rotation_to_transform, quaternion_matrix_batched_array, quaternion_from_matrix_batched_array


def random_rotation_matrix(shape=None, seed=None, as_tensor=False):
    """
    Returns a SO(3) rotation matrix drawn from the Haar distribution
    (the only uniform distribution on SO(N)) with a determinant of +1.
    Args:
        shape ():
        seed ():
    Returns:
        rotation_matrices (): of shape (..., 3, 3)
    """
    if shape is None:
        num_rots = 1
        out_shape = (3, 3)
    elif type(shape) in [int]:
        num_rots = shape
        out_shape = (shape, 3, 3)
    else:
        shape = np.asarray(shape)
        num_rots = np.prod(shape)
        out_shape = list(shape) + [3, 3]
    rotation_matrices = special_ortho_group.rvs(dim=3, size=num_rots, random_state=seed)
    # reshape it
    rotation_matrices = rotation_matrices.reshape(out_shape) # (..., 3, 3)
    # rotation to transform
    if as_tensor:
        rotation_matrices = torch.tensor(rotation_matrices, dtype=torch.float32)
    return rotation_matrices


def random_transform_matrix(shape=None, seed=None, as_tensor=False):
    random_rotation_matrices = random_rotation_matrix(shape=shape, seed=seed, as_tensor=as_tensor)
    random_transform_matrices = rotation_to_transform(random_rotation_matrices)
    return random_transform_matrices


def random_quaternion(shape=None, seed=None, as_tensor=False):
    """
    Return a random unit quaternion.
    """
    random_transform_matrices = random_transform_matrix(shape=shape, seed=seed, as_tensor=as_tensor)
    random_quat = quaternion_from_matrix_batched_array(random_transform_matrices)
    return random_quat


# DEBUG:
if __name__ == '__main__':
    X = random_rotation_matrix(shape=(10, 20))
    print(X.shape)
    random_quat = random_quaternion(shape=(10, 20))
    print(random_quat.shape)
