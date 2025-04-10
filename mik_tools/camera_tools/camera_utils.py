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


def change_resolution_camera_matrix(K:np.ndarray, img_size_new, img_size_old=None) -> np.ndarray:
    """
    Preserve the field of view but increase resolution
    :param K: (..., 3, 3)
    :param img_size_new: (..., 2)
    :param img_size_old: (..., 2) if None, use the current K
    :return:
    """
    if img_size_old is None:
        img_size_old = 2 * K[..., :2, -1] # (..., 2)
    K_new = K.copy()
    K_new[..., 0, 0] = K[..., 0, 0] * img_size_new[0] / img_size_old[0]
    K_new[..., 1, 1] = K[..., 1, 1] * img_size_new[1] / img_size_old[1]
    K_new[..., 0, 2] = img_size_new[0] * 0.5
    K_new[..., 1, 2] = img_size_new[1] * 0.5
    return K_new


def modify_img_size_camera_matrix(K:np.ndarray, img_size_new) -> np.ndarray:
    """
    This updates the image size which is equivalent of cropping or padding the image
    This operation preserves the spatial resolution of the camera.
    If image_size_new is smaller than the original image size, it is like rendering a SUBSET of the image
    :param K: (..., 3, 3)
    :param img_size_new: (..., 2)
    :param img_size_old: (..., 2) if None, use the current K
    :return:
    """
    K_new = K.copy()
    K_new[..., :2, -1] = np.floor(img_size_new * 0.5).astype(np.int64) # (..., 2)
    return K_new
