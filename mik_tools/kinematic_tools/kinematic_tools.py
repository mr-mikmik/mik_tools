import numpy as np
import torch

from mik_tools import tr, matrix_to_pose, pose_to_matrix, matrix_to_pose_2d, pose_to_matrix_2d, transform_matrix_inverse
from mik_tools.mik_tf_tools.tr_trajectory_utils import project_pose_to_plane, unproject_planar_pose, get_w_T_plane


def skew_matrix(v):
    """
    Get the skew matrix of a vector.
    :param v: R^3 vector. of shape (..., 3) as a list or numpy array or torch tensor
    :return: skew matrix of shape (..., 3, 3)
    """
    if isinstance(v, list):
        v = np.array(v)
        skew_matrix = skew_matrix_array(v)
    elif isinstance(v, np.ndarray):
        skew_matrix = skew_matrix_array(v)
    elif isinstance(v, torch.Tensor):
        skew_matrix = skew_matrix_tensor(v)
    else:
        raise ValueError('v must be a list or a numpy array.')
    return skew_matrix


def get_adjoint_matrix(T=None, R=None, t=None):
    """
    Get the adjoint matrix of a transformation matrix.
    :param T: Homogeneous transfomration matrix. (..., 4, 4) numpy array
    :param R: Rotation matrix. (..., 3, 3) numpy array or torch tensor
    :param t: Translation vector. (..., 3) numpy array or torch tensor. If not provided, it is assumed to be zero.
    :return: Adjoint matrix. (..., 6, 6) numpy array
    The adjoint matrix is defined as the matrix that converts twists as follows
    given a twist nu_B = [v_B, omega_B]^T in frame B, the twist nu_A = [v_A, omega_A]^T in frame A is given by
    nu_A = Ad(A_T_B) nu_B, where Ad(A_T_B) is the adjoint matrix of the transformation matrix A_T_B
    This matrix is defined as:
        Ad(A_T_B) = [[A_R_B, 0], [[A_t_B^] A_R_B, A_R_B]]
        where R is the rotation matrix and t is the translation vector and [t^] is the skew matrix of t
    Note: For wrench transformations, the adjoint matrix is used as follows:
        wrench_f2 = [Ad(f1_X_f2)]^T wrench_f1
        Note that R^T = R^-1 so A_R_B^T = B_R_A
    """
    if isinstance(T, np.ndarray) or isinstance(R, np.ndarray):
        adjoint_matrix = get_adjoint_matrix_numpy(R=R, t=t, T=T)
    elif isinstance(T, torch.Tensor) or isinstance(R, torch.Tensor):
        adjoint_matrix = get_adjoint_matrix_torch(R=R, t=t, T=T)
    else:
        raise ValueError('R or T must be a numpy array or a torch tensor.')
    return adjoint_matrix


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


def transform_twist(twist_f1, f2_X_f1):
    """
    Transform a twist from frame f1 to frame f2.
    :param twist_f1: numpy or torch tensor of shape (..., 6) with the twist in the frame f1
    :param f2_X_f1: numpy or torch tensor of shape (..., 4, 4) with the transformation matrix
    :return: twist_f2: numpy or torch tensor of shape (..., 6) with the twist in the frame f2

    NOTE:
    The twist is a 6D vector [v_x, v_y, v_z, omega_x, omega_y, omega_z] in R^6
    the transformation is done as follows:
    twist_f2 = [Ad(f2_X_f1)] twist_f1
    where twist_f2 = [v_x, v_y, v_z, omega_x, omega_y, omega_z]_f2 and
          twist_f1 = [v_x, v_y, v_z, omega_x, omega_y, omega_z]_f1
    """
    f2_Ad_f1 = get_adjoint_matrix(T=f2_X_f1)  # (6x6) matrix (..., 6, 6)
    if isinstance(twist_f1, list):
        twist_f1 = np.array(twist_f1)
        twist_f2 = f2_Ad_f1 @ twist_f1
    elif isinstance(twist_f1, np.ndarray):
        twist_f2 = np.einsum('...ij,...j->...i', f2_Ad_f1, twist_f1)
    elif isinstance(twist_f1, torch.Tensor):
        twist_f2 = torch.einsum('...ij,...j->...i', f2_Ad_f1, twist_f1)
    else:
        raise ValueError('twist_f1 must be a list or a numpy array.')
    return twist_f2


def transform_wrench(wrench_wf, wf_T_tf):
    """
    Transform a wrench from the wrench frame to a target frame.
    :param wrench_wf: numpy or torch tensor of shape (..., 6) with the wrench in the wrench frame
    :param wf_T_tf: transformation matrix from the wrench frame to the target frame of shape (..., 4, 4)
    :return: wrench_tf: numpy or torch tensor of shape (..., 6) with the wrench in the target frame

    Note: wrench_f2 = [Ad(f1_X_f2)]^T wrench_f1
    in other words:
        wrench_f2 = [            f2_R_f1,       0] wrench_f1
                    [ f2_R_f1 [f2_t_f1^] , f2_R_f1]
        where wrench_f1 = [Fx, Fy, Fz, Tx, Ty, Tz]_f1
    """
    wf_Ad_tf = get_adjoint_matrix(T=wf_T_tf)  # (6x6) matrix (..., 6, 6)
    if isinstance(wrench_wf, list):
        wrench_wf = np.array(wrench_wf)
        wrench_tf = wf_Ad_tf.T @ wrench_wf
    elif isinstance(wrench_wf, np.ndarray):
        wrench_tf = np.einsum('...ji,...j->...i', wf_Ad_tf, wrench_wf) # tf_ad_wf.T @ wrench_wf
    elif isinstance(wrench_wf, torch.Tensor):
        wrench_tf = torch.einsum('...ji,...j->...i', wf_Ad_tf, wrench_wf)
    else:
        raise ValueError('wrench_wf must be a list or a numpy array.')
    return wrench_tf


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



# ==================================================================================================
# ==================================================================================================
# AUXILIARY FUNCTIONS
# ==================================================================================================

def skew_matrix_array(v):
    """
    Get the skew matrix of a vector.
    :param v: R^3 vector. as numpy array of shape (..., 3)
    :return: skey matrix of shape (..., 3, 3)
    """
    skew_matrix = np.zeros(v.shape[:-1] + (3, 3))
    skew_matrix[..., 0, 1] = -v[..., 2]
    skew_matrix[..., 0, 2] = v[..., 1]
    skew_matrix[..., 1, 0] = v[..., 2]
    skew_matrix[..., 1, 2] = -v[..., 0]
    skew_matrix[..., 2, 0] = -v[..., 1]
    skew_matrix[..., 2, 1] = v[..., 0]
    return skew_matrix


def skew_matrix_tensor(v):
    """
    Get the skew matrix of a vector.
    :param v: R^3 vector. as a torch tensor of shape (..., 3)
    :return:
    """
    skew_matrix = torch.zeros(v.shape[:-1] + (3, 3), dtype=v.dtype, device=v.device)
    skew_matrix[..., 0, 1] = -v[..., 2]
    skew_matrix[..., 0, 2] = v[..., 1]
    skew_matrix[..., 1, 0] = v[..., 2]
    skew_matrix[..., 1, 2] = -v[..., 0]
    skew_matrix[..., 2, 0] = -v[..., 1]
    skew_matrix[..., 2, 1] = v[..., 0]
    return skew_matrix


def get_adjoint_matrix_numpy(R=None, t=None, T=None):
    """
    Get the adjoint matrix of a transformation matrix.

    :param R: Rotation matrix. (..., 3, 3) numpy array
    :param t: Translation vector. (..., 3) numpy array
    :param T: Homogeneous transfomration matrix. (..., 4, 4) numpy array
    :return: Adjoint matrix. (..., 6, 6) numpy array

    The adjoint matrix is defined as the matrix that converts twists as follows
    given a twist nu_B = [v_B, omega_B]^T in frame B, the twist nu_A = [v_A, omega_A]^T in frame A is given by
    nu_A = Ad(A_T_B) nu_B, where Ad(A_T_B) is the adjoint matrix of the transformation matrix A_T_B
    This matrix is defined as:
    Ad(A_T_B) = [[ A_R_B, 0 ],
                 [ [A_t_B^] A_R_B, A_R_B]]
    where R is the rotation matrix and t is the translation vector and [t^] is the skew matrix of t
    Note: For wrench transformations, the adjoint matrix is used as follows:
        wrench_f2 = [Ad(f1_X_f2)]^T wrench_f1
        Note that R^T = R^-1 so A_R_B^T = B_R_A
        where the wrench is a 6D vector [fx, fy, fz, tau_x, tau_y, tau_z] in R^6
    """
    if T is not None:
        R = T[..., :3,:3]
        t = T[..., :3,3]
    elif t is None:
        t = np.zeros(R.shape[:-2] + (3,))
    else:
        raise ValueError("Either (R and t) or T must be provided.")
    batch_size = R.shape[:-2]
    adjoint_matrix = np.zeros(batch_size + (6, 6))
    adjoint_matrix[..., :3, :3] = R
    adjoint_matrix[..., 3:, 3:] = R
    adjoint_matrix[..., :3, 3:] = np.einsum('...ij,...jk->...ik', skew_matrix(t), R)
    return adjoint_matrix


def get_adjoint_matrix_torch(R=None, t=None, T=None):
    """
    Get the adjoint matrix of a transformation matrix.

    :param R: Rotation matrix. (..., 3, 3) torch tensor
    :param t: Translation vector. (..., 3) torch tensor
    :param R: Homogeneous transfomration matrix. (..., 4, 4) torch tensor
    :return: Adjoint matrix. (..., 6, 6) torch tensor
    The adjoint matrix is defined as the matrix that converts twists as follows
    given a twist nu_B = [v_B, omega_B]^T in frame B, the twist nu_A = [v_A, omega_A]^T in frame A is given by
    nu_A = Ad(A_T_B) nu_B, where Ad(A_T_B) is the adjoint matrix of the transformation matrix A_T_B
    This matrix is defined as:
    Ad(A_T_B) = [[A_R_B, 0], [[A_t_B^] A_R_B, A_R_B]]
    where R is the rotation matrix and t is the translation vector and [t^] is the skew matrix of t
    """
    if T is not None:
        R = T[..., :3,:3]
        t = T[..., :3,3]
    elif t is None:
        t = torch.zeros(R.shape[:-2] + (3,), dtype=R.dtype, device=R.device)
    else:
        raise ValueError("Either (R and t) or T must be provided.")
    batch_size = R.shape[:-2]
    adjoint_matrix = torch.zeros(batch_size + (6, 6), dtype=R.dtype, device=R.device)
    adjoint_matrix[..., :3, :3] = R
    adjoint_matrix[..., 3:, 3:] = R
    adjoint_matrix[..., :3, 3:] = torch.einsum('...ij,...jk->...ik', skew_matrix(t), R)
    return adjoint_matrix


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

# ==================================================================================================





# TESTS AND DEBUG:
if __name__ == '__main__':

    # # TEST WITHOUT ROTATIONS ------------------------
    # # test pose transfomations
    # plane = np.array([0, 0, 1])
    # point = np.array([1, 1, 0])
    # w_T_plane = get_w_T_plane(plane=plane, point=point)
    # w_T_wf = pose_to_matrix(np.array([4, 3, 1, 0., 0, 0, 1]))
    # plane_T_wf = np.linalg.inv(w_T_plane) @ w_T_wf
    # wf_T_plane = np.linalg.inv(plane_T_wf)
    # print("w_T_plane: ", w_T_plane)
    # print("plane_T_wf: ", plane_T_wf)
    #
    # # CASE 1 -
    # print('CASE 1')
    # wrench_wf = np.array([1, 0, 0, 0, 0, 0])
    # wrench_pf = transform_wrench(wrench_wf, wf_T_tf=wf_T_plane)
    # print("wrench_wf: ", wrench_wf)
    # print("wrench_pf: ", wrench_pf)
    # projected_wrench_wf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf)
    # projected_wrench_pf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf, plane_frame_coords=True)
    # print("projected_wrench_pf: ", projected_wrench_pf)
    # print("projected_wrench_wf: ", projected_wrench_wf)
    # projected_wrench_wf_2 = project_wrench_to_plane_2(wrench_wf, w_T_plane, w_T_wf)
    # print("projected_wrench_wf_2: ", projected_wrench_wf_2)
    # # CASE 1 -
    # print('CASE 2')
    # wrench_wf = np.array([1, 2, 3, 0, 0, 0])
    # wrench_pf = transform_wrench(wrench_wf, wf_T_tf=wf_T_plane)
    # print("wrench_wf: ", wrench_wf)
    # print("wrench_pf: ", wrench_pf)
    # projected_wrench_wf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf)
    # projected_wrench_pf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf, plane_frame_coords=True)
    # print("projected_wrench_pf: ", projected_wrench_pf)
    # print("projected_wrench_wf: ", projected_wrench_wf)
    # projected_wrench_wf_2 = project_wrench_to_plane_2(wrench_wf, w_T_plane, w_T_wf)
    # print("projected_wrench_wf_2: ", projected_wrench_wf_2)
    # # CASE 3 -
    # print('CASE 3')
    # wrench_wf = np.array([6, 5, 4, 3, 2, 1])
    # wrench_pf = transform_wrench(wrench_wf, wf_T_tf=wf_T_plane)
    # print("wrench_wf: ", wrench_wf)
    # print("wrench_pf: ", wrench_pf)
    # projected_wrench_wf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf)
    # projected_wrench_pf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf, plane_frame_coords=True)
    # print("projected_wrench_pf: ", projected_wrench_pf)
    # print("projected_wrench_wf: ", projected_wrench_wf)
    # projected_wrench_wf_2 = project_wrench_to_plane_2(wrench_wf, w_T_plane, w_T_wf)
    # print("projected_wrench_wf_2: ", projected_wrench_wf_2)

    # TEST ROTATIONS ------------------------
    # CASE 4 - --- Test with rotations on z axis
    plane = np.array([0, 0, 1])
    point = np.array([1, 1, 0])
    w_T_plane = get_w_T_plane(plane=plane, point=point)
    w_T_wf = pose_to_matrix(np.concatenate([np.array([4, 3, 1]), tr.quaternion_about_axis(np.pi/2, [0, 0, 1])])) # 90 degree rotation
    plane_T_wf = np.linalg.inv(w_T_plane) @ w_T_wf
    wf_T_plane = np.linalg.inv(plane_T_wf)
    print("w_T_plane: ", w_T_plane)
    print("plane_T_wf: ", plane_T_wf)
    print('CASE 4')
    wrench_wf = np.array([1, 0, 0, 0, 0, 0])
    wrench_pf = transform_wrench(wrench_wf, wf_T_tf=wf_T_plane)
    print("wrench_wf: ", wrench_wf)
    print("wrench_pf: ", wrench_pf)
    projected_wrench_wf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf)
    projected_wrench_pf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf, plane_frame_coords=True)
    print("projected_wrench_pf: ", projected_wrench_pf)
    print("projected_wrench_wf: ", projected_wrench_wf)
    projected_wrench_wf_2 = project_wrench_to_plane_2(wrench_wf, w_T_plane, w_T_wf)
    print("projected_wrench_wf_2: ", projected_wrench_wf_2)
    print('CASE 5')
    wrench_wf = np.array([0, 1, 0, 0, 0, 0])
    wrench_pf = transform_wrench(wrench_wf, wf_T_tf=wf_T_plane)
    print("wrench_wf: ", wrench_wf)
    print("wrench_pf: ", wrench_pf)
    projected_wrench_wf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf)
    projected_wrench_pf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf, plane_frame_coords=True)
    print("projected_wrench_pf: ", projected_wrench_pf)
    print("projected_wrench_wf: ", projected_wrench_wf)
    projected_wrench_wf_2 = project_wrench_to_plane_2(wrench_wf, w_T_plane, w_T_wf)
    print("projected_wrench_wf_2: ", projected_wrench_wf_2)
    print('CASE 6')
    wrench_wf = np.array([1, 1, 0, 0, 0, 1])
    wrench_pf = transform_wrench(wrench_wf, wf_T_tf=wf_T_plane)
    print("wrench_wf: ", wrench_wf)
    print("wrench_pf: ", wrench_pf)
    projected_wrench_wf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf)
    projected_wrench_pf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf, plane_frame_coords=True)
    print("projected_wrench_pf: ", projected_wrench_pf)
    print("projected_wrench_wf: ", projected_wrench_wf)
    projected_wrench_wf_2 = project_wrench_to_plane_2(wrench_wf, w_T_plane, w_T_wf)
    print("projected_wrench_wf_2: ", projected_wrench_wf_2)
    print('CASE 7')
    wrench_wf = np.array([1, 1, 1, 1, 1, 1])
    wrench_pf = transform_wrench(wrench_wf, wf_T_tf=wf_T_plane)
    print("wrench_wf: ", wrench_wf)
    print("wrench_pf: ", wrench_pf)
    projected_wrench_wf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf)
    projected_wrench_pf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf, plane_frame_coords=True)
    print("projected_wrench_pf: ", projected_wrench_pf)
    print("projected_wrench_wf: ", projected_wrench_wf)
    projected_wrench_wf_2 = project_wrench_to_plane_2(wrench_wf, w_T_plane, w_T_wf)
    print("projected_wrench_wf_2: ", projected_wrench_wf_2)

    print('\n\n')

    # TEST ROTATIONS ------------------------
    # CASE 8 - --- Test with rotations
    plane = np.array([0, -1, 0])
    point = np.array([1, 0, 1])
    w_T_plane = get_w_T_plane(plane=plane, point=point)
    w_T_wf = pose_to_matrix(
        np.concatenate([np.array([4, 3, 1]), tr.quaternion_about_axis(0, [0, 0, 1])]))  # 90 degree rotation
    plane_T_wf = np.linalg.inv(w_T_plane) @ w_T_wf
    wf_T_plane = np.linalg.inv(plane_T_wf)
    print("w_T_plane: ", w_T_plane)
    print("plane_T_wf: ", plane_T_wf)
    print('CASE 8')
    wrench_wf = np.array([1, 0, 0, 0, 0, 0])
    wrench_pf = transform_wrench(wrench_wf, wf_T_tf=wf_T_plane)
    print("wrench_wf: ", wrench_wf)
    print("wrench_pf: ", wrench_pf)
    projected_wrench_wf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf)
    projected_wrench_pf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf, plane_frame_coords=True)
    print("projected_wrench_pf: ", projected_wrench_pf)
    print("projected_wrench_wf: ", projected_wrench_wf)
    projected_wrench_wf_2 = project_wrench_to_plane_2(wrench_wf, w_T_plane, w_T_wf)
    print("projected_wrench_wf_2: ", projected_wrench_wf_2)

    print('CASE 9')
    wrench_wf = np.array([0, 1, 0, 0, 0, 0])
    wrench_pf = transform_wrench(wrench_wf, wf_T_tf=wf_T_plane)
    print("wrench_wf: ", wrench_wf)
    print("wrench_pf: ", wrench_pf)
    projected_wrench_wf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf)
    projected_wrench_pf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf, plane_frame_coords=True)
    print("projected_wrench_pf: ", projected_wrench_pf)
    print("projected_wrench_wf: ", projected_wrench_wf)
    projected_wrench_wf_2 = project_wrench_to_plane_2(wrench_wf, w_T_plane, w_T_wf)
    print("projected_wrench_wf_2: ", projected_wrench_wf_2)

    print('CASE 10')
    wrench_wf = np.array([1, 1, 0, 0, 0, 1])
    wrench_pf = transform_wrench(wrench_wf, wf_T_tf=wf_T_plane)
    print("wrench_wf: ", wrench_wf)
    print("wrench_pf: ", wrench_pf)
    projected_wrench_wf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf)
    projected_wrench_pf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf, plane_frame_coords=True)
    print("projected_wrench_pf: ", projected_wrench_pf)
    print("projected_wrench_wf: ", projected_wrench_wf)
    projected_wrench_wf_2 = project_wrench_to_plane_2(wrench_wf, w_T_plane, w_T_wf)
    print("projected_wrench_wf_2: ", projected_wrench_wf_2)

    print('CASE 11')
    wrench_wf = np.array([1, 1, 1, 1, 1, 1])
    wrench_pf = transform_wrench(wrench_wf, wf_T_tf=wf_T_plane)
    print("wrench_wf: ", wrench_wf)
    print("wrench_pf: ", wrench_pf)
    projected_wrench_wf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf)
    projected_wrench_pf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf, plane_frame_coords=True)
    print("projected_wrench_pf: ", projected_wrench_pf)
    print("projected_wrench_wf: ", projected_wrench_wf)
    projected_wrench_wf_2 = project_wrench_to_plane_2(wrench_wf, w_T_plane, w_T_wf)
    print("projected_wrench_wf_2: ", projected_wrench_wf_2)

