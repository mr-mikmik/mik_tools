import numpy as np
from mik_tools import matrix_to_pose


def compute_camera_pose(position:np.ndarray, lookat:np.ndarray, view_up=np.array([0, 0, 1]), as_matrix=False):
    """
    Compute the camera pose given the position, lookat and view_up vectors
    Args:
        position (np.ndarray): (3,) array representing the camera position
        lookat (np.ndarray): (3,) array representing the camera lookat point
        view_up (np.ndarray): (3,) array representing the camera view up vector
    Returns:
        np.ndarray: (4,4) array representing the camera pose
    """
    z_axis = lookat - position
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(view_up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    R = np.array([x_axis, y_axis, z_axis]).T
    t = position
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t
    if as_matrix:
        return pose
    else:
        return matrix_to_pose(pose)