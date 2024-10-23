import numpy as np
import torch

from mik_tools import matrix_to_pose, pose_to_matrix_2d
from mik_tools.kinematic_tools import transform_wrench
from mik_tools.mik_tf_tools.tr_trajectory_utils import project_pose_to_plane


def get_adjoint_matrix_planar(R=None, t=None, T=None):
    """
    Get the adjoint matrix of a transformation matrix.

    :param R: Rotation matrix.
    :param t: Translation vector.
    :return: Adjoint matrix.
    """
    if T is not None:
        R = T[:2,:2]
        t = T[:2,2]
    elif t is None:
        t = np.zeros(2)
    else:
        raise ValueError("Either (R and t) or T must be provided.")
    adjoint_matrix = np.zeros((3, 3))
    adjoint_matrix[:2, :2] = R
    adjoint_matrix[2, 2] = 1
    adjoint_matrix[:2, 2] = np.array([-t[1], t[0]])@R # skew operation to do the cross product
    return adjoint_matrix


def transform_wrench_planar(wrench_wf, wf_T_tf):
    """
    Transform a wrench from the wrench frame to a target frame.
    :param wrench_wf:
    :param wf_T_tf: transformation matrix from the wrench frame to the target frame
    :return:
    """
    # tf_Ad_wf = get_adjoint_matrix_planar(T=wf_T_tf_2d)  # (3x3) matrix
    # print('tf_Ad_wf.T: ', tf_Ad_wf.T)
    # wrench_tf = tf_Ad_wf.T @ wrench_wf  # wrench at target frame expressed in plane_coordinates
    # R = wf_T_tf_2d[:2,:2]
    # t = wf_T_tf_2d[:2,2]
    # tx, ty = t
    # A = np.eye(3)
    # A[2,:2] = np.array([ty, -tx])
    # B = np.eye(3)
    # B[:2,:2] = R
    # wrench_tf = np.linalg.inv(B) @ A @ wrench_wf
    # return wrench_tf
    if isinstance(wrench_wf, list):
        wrench_wf = np.array(wrench_wf)
        wrench_tf = transform_wrench_planar_array(wrench_wf, wf_T_tf)
    elif isinstance(wrench_wf, np.ndarray):
        wrench_tf = transform_wrench_planar_array(wrench_wf, wf_T_tf)
    elif isinstance(wrench_wf, torch.Tensor):
        wrench_tf = transform_wrench_planar_tensor(wrench_wf, wf_T_tf)
    else:
        raise ValueError('wrench_wf must be a list or a numpy array.')
    return wrench_tf


def project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf, ff_quat=None, gw_axis=None, plane_frame_coords=False):
    """
    Project a wrench to a plane.
    :param wrench: Wrench in the world frame. as a 6D vector [fx, fy, fz, tau_x, tau_y, tau_z]
    :param plane: Plane normal vector in the world frame.
    :return: Projected wrench.
    # the projected wrench is the wrench in the plane coordinates [fx, fy, tau]
    """

    # transform the wrench in the plane coordinates
    wf_X_plane = np.linalg.inv(w_T_wf) @ w_T_plane
    wrench_plane = transform_wrench(wrench_wf, wf_X_plane) # wrench in the plane_coordinates
    # project the wrench cooridnates into plane coordinates
    projected_wrench_pf = np.array([wrench_plane[0], wrench_plane[1], wrench_plane[5]]) # (3,) vector composed by [fx, fy, tau]
    # transform the
    if plane_frame_coords:
        return projected_wrench_pf
    else:
        # finally, transform the wrench back to the coordinates of the wrench frame but in its projection to planar frame
        w_pose_wf = matrix_to_pose(w_T_wf)
        pf_pose_wf_planar = project_pose_to_plane(w_pose_wf, w_T_plane, ff_quat=ff_quat, gw_axis=gw_axis)
        pf_X_wf_2d = pose_to_matrix_2d(pf_pose_wf_planar)
        wf_X_pf_2d = np.linalg.inv(pf_X_wf_2d)
        projected_wrench_wf = transform_wrench_planar(projected_wrench_pf, pf_X_wf_2d)
        return projected_wrench_wf


def project_wrench_to_plane_2(wrench_wf, w_T_plane, w_T_wf, ff_quat=None, gw_axis=None):
    """
    Project a wrench to a plane.
    :param wrench: Wrench in the world frame. as a 6D vector [fx, fy, fz, tau_x, tau_y, tau_z]
    :param plane: Plane normal vector in the world frame.
    :return: Projected wrench.
    # the projected wrench is the wrench in the plane coordinates [fx, fy, tau]

    NOTE: This function is equivalent according to the tests to projecte_wrench_to_plane
    """

    # transform the wrench in the plane coordinates
    wf_X_plane = np.linalg.inv(w_T_wf) @ w_T_plane
    plane_X_wf = np.linalg.inv(wf_X_plane)
    wf_X_wf_plane_oriented = wf_X_plane
    wf_X_wf_plane_oriented[:3, 3] = 0 # only orientation, no translation
    wf_plane_oriented_X_wf = np.linalg.inv(wf_X_wf_plane_oriented)
    wrench_wf_plane_oriented = transform_wrench(wrench_wf, wf_X_wf_plane_oriented) # wrench in the plane_coordinates
    # project the wrench cooridnates into plane coordinates
    projected_wrench_wf_plane_oriented = np.array([wrench_wf_plane_oriented[0], wrench_wf_plane_oriented[1], wrench_wf_plane_oriented[5]]) # (3,) vector composed by [fx, fy, tau]
    # need now to express it on the right plane coordinates
    w_pose_wf = matrix_to_pose(w_T_wf)
    pf_pose_wf_planar = project_pose_to_plane(w_pose_wf, w_T_plane, ff_quat=ff_quat, gw_axis=gw_axis)
    pf_X_wf_2d = pose_to_matrix_2d(pf_pose_wf_planar)
    wf_plane_oriented_X_wf_2d = pf_X_wf_2d.copy()
    wf_plane_oriented_X_wf_2d[:2, 2] = 0 # only orientation, no translation
    projected_wrench_wf = transform_wrench_planar(projected_wrench_wf_plane_oriented, wf_plane_oriented_X_wf_2d)
    # import pdb; pdb.set_trace()
    return projected_wrench_wf


def unproject_wrench_to_plane(planar_wrench, w_T_plane, w_T_wf, ff_quat=None, gw_axis=None):
    """
    Unproject a wrench from a plane.
    :param planar_wrench:
    :param w_T_plane:
    :param w_T_wf:
    :param ff_quat:
    :param gw_axis:
    :return:
    """
    # transform the wrench in the plane coordinates
    wf_X_plane = np.linalg.inv(w_T_wf) @ w_T_plane
    w_pose_wf = matrix_to_pose(w_T_wf)
    pf_pose_wf_planar = project_pose_to_plane(w_pose_wf, w_T_plane, ff_quat=ff_quat, gw_axis=gw_axis)
    pf_X_wf_2d = pose_to_matrix_2d(pf_pose_wf_planar)
    projected_wrench_wf = planar_wrench
    projected_wrench_pf = transform_wrench_planar(projected_wrench_wf, np.linalg.inv(pf_X_wf_2d))
    # project the wrench from plane_coordinates to world coordinates
    wrench_pf = np.array([projected_wrench_pf[0], projected_wrench_pf[1], 0, 0, 0, projected_wrench_pf[2]]) # wrench in the plane_coordinates
    # finally, transform the wrench back to the coordinates of the wrench frame
    wrench_wf = transform_wrench(wrench_pf, wf_X_plane)
    return wrench_wf


def transform_wrench_planar_array(wrench_wf_array, wf_T_tf):
    """
    Transform the planar wrench from wf to tf
    :param wrench_wf_array: (...,  3)
    :param wf_T_tf: (..., 3, 3)
    :return: (..., 3)
    """
    # tf_Ad_wf = get_adjoint_matrix_planar(T=wf_T_tf_2d)  # (3x3) matrix
    # print('tf_Ad_wf.T: ', tf_Ad_wf.T)
    # wrench_tf = tf_Ad_wf.T @ wrench_wf  # wrench at target frame expressed in plane_coordinates
    R = wf_T_tf[...,:2,:2]
    t = wf_T_tf[...,:2,2]
    tx, ty = t[...,0], t[...,1]
    # Create A as a (..., 3, 3) eye matrix
    A = np.zeros(wf_T_tf.shape[:-2] + (3, 3))
    A[...,0,0] = 1
    A[...,1,1] = 1
    A[...,2,2] = 1
    a = np.stack([ty, -tx], axis=-1) # FIX THIS
    A[..., 2, :2] = a
    B = np.zeros(wf_T_tf.shape[:-2] + (3, 3))
    B[..., :2,:2] = R
    B[...,2,2] = 1
    B_inv = np.linalg.inv(B)
    wrench_tf_array = np.einsum('...ij,...jk,...k->...i', B_inv, A, wrench_wf_array)
    return wrench_tf_array


def transform_wrench_planar_tensor(wrench_wf_tensor, wf_T_tf):
    """
    Transform the planar wrench from wf to tf
    :param wrench_wf_tensor: (...,  3) torch tensor
    :param wf_T_tf: (..., 3, 3) torch tensor
    :return: (..., 3) torch tensor
    """
    R = wf_T_tf[..., :2, :2]
    t = wf_T_tf[..., :2, 2]
    tx, ty = t[..., 0], t[..., 1]
    # Create A as a (..., 3, 3) eye matrix
    A = torch.zeros(wf_T_tf.shape[:-2] + (3, 3), dtype=wrench_wf_tensor.dtype, device=wrench_wf_tensor.device)
    A[..., 0, 0] = 1
    A[..., 1, 1] = 1
    A[..., 2, 2] = 1
    a = torch.stack([ty, -tx], dim=-1)  # FIX THIS
    A[..., 2, :2] = a
    B = torch.zeros(wf_T_tf.shape[:-2] + (3, 3), dtype=wrench_wf_tensor.dtype, device=wrench_wf_tensor.device)
    B[..., :2, :2] = R
    B[..., 2, 2] = 1
    B_inv = torch.linalg.inv(B)
    wrench_tf_array = torch.einsum('...ij,...jk,...k->...i', B_inv, A, wrench_wf_tensor)
    return wrench_tf_array
