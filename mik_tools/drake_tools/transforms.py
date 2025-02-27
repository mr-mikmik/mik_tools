try:
    from pydrake.math import RigidTransform, RollPitchYaw
except ImportError:
    pass


def matrix_to_rigid_transform(matrix):
    """
    Convert a 4x4 matrix to a RigidTransform
    :param matrix: (4, 4) as the matrix
    :return: rigid_transform: RigidTransform as the rigid transform
    """
    return RigidTransform(matrix)