import numpy as np
import torch
from scipy.linalg import expm
from typing import List, Tuple, Union

from mik_tools import tr, pose_to_matrix, transform_matrix_inverse


def vector_to_skew_matrix(v:Union[List, np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Get the skew matrix of a vector. v = [v_x, v_y, v_z]
    This skew symmetric matrix belongs to the so(3) group. (exp:so(3)->SO(3))
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


def skew_matrix_to_vector(skew_matrix:Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Get the vector from a skew symmetric matrix.
    :param skew_matrix: (..., 3, 3) skew symmetric matrix
    :return: vector of shape (..., 3)
    """
    v1 = skew_matrix[..., 2, 1]
    v2 = skew_matrix[..., 0, 2]
    v3 = skew_matrix[..., 1, 0]
    vs = [v1, v2, v3]
    if isinstance(skew_matrix, np.ndarray):
        v = np.stack(vs, axis=-1) # (..., 3)
    elif isinstance(skew_matrix, torch.Tensor):
        v = torch.stack(vs, dim=-1) # (..., 3)
    else:
        raise ValueError(f'skew_matrix must be a numpy array or a torch tensor. Got {type(skew_matrix)}')
    return v


def exponential_map(skew_matrix:Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Get the exponential map of a matrix.
    :param skew_matrix: (..., 3, 3) skew symmetric matrix in so(3)
    :return: exponential map of shape (..., 3, 3) in SO(3)
    The exponential map of a skew symmetric matrix X is defined as:
    exp(X) = I + X + X^2/2! + X^3/3! + X^4/4! + ...
    Using the Rodrigues formula, the exponential map of a skew symmetric matrix X is given by:
    exp(X) = I + sin(theta)X + (1 - cos(theta))X^2
    where theta = ||X|| is the angle of rotation
    NOTE: The exponential map is used to convert a skew symmetric matrix to a rotation matrix. In other words:
        exp: so(3) -> SO(3)  where  R = exp(X) where X is a skew symmetric matrix
    """
    if isinstance(skew_matrix, np.ndarray):
        # theta = np.linalg.norm(skew_matrix_to_vector(skew_matrix), axis=-1, keepdims=True)  # (..., 1)
        # skew_matrix_squared = np.einsum('...ij,...jk->...ik', skew_matrix, skew_matrix)  # (..., 3, 3)
        # exp_X = np.eye(3) + np.sin(theta) * skew_matrix + (1 - np.cos(theta)) * skew_matrix_squared # (..., 3, 3)
        exp_X = expm(skew_matrix) # (..., 3, 3)
    elif isinstance(skew_matrix, torch.Tensor):
        # rotation_vector = skew_matrix_to_vector(skew_matrix) # (..., 3)
        # theta = torch.norm(skew_matrix_to_vector(skew_matrix), dim=-1, keepdim=True) # (..., 1)
        # # NOTE: For small thetas, we can use the Taylor series expansion
        # # exp(skew_matrix) ≈ I + skew_matrix
        # axis = rotation_vector / theta  # (..., 3)
        # axis_skew = vector_to_skew_matrix(axis) # (..., 3, 3)
        # axis_skew_squared = torch.einsum('...ij,...jk->...ik', axis_skew, axis_skew)  # (..., 3, 3)
        # exp_X = torch.eye(3, device=skew_matrix.device, dtype=skew_matrix.dtype) + torch.sin(theta) * axis_skew + (1 - torch.cos(theta)) * axis_skew_squared # (..., 3, 3)
        exp_X = torch.matrix_exp(skew_matrix)
    else:
        raise ValueError(f'skew_matrix must be a numpy array or a torch tensor. Got {type(skew_matrix)}')
    return exp_X


def get_adjoint_matrix(T=None, R=None, t=None) -> Union[np.ndarray, torch.Tensor]:
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


def get_contact_jacobian(wf_X_cf:Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Return the jacobian associated to the contact frame cf in the wrench frame wf
    :param wf_X_cf: (..., 4, 4) transformation from contact frame to wrench frame where the jacobian is computed
    :return: Jc (..., 6, 3) jacobian associated to the contact frame cf in the wrench frame wf
    NOTE: we assume that contact forces are given as [fn, ft_1, ft_2].T where fn is the normal force and ft_1, ft_2 are the tangential forces
    Therefore the jacobian is computed as:
    Jc = [R.T | [^pc].T].T
    where R is the rotation matrix from the contact frame to the wrench frame
    and pc is the vector from the contact frame to the wrench
    wrench_contact_wf = [fx, fy, fz, tau_x, tau_y, tau_z]_wf.T = Jc @ wrench_contact_cf = Jc @ [fn, ft_1, ft_2]_cf.T
    """
    # wf_X_cf: transfromation from contact frame to wrench frame
    # Jc: [R.T | [^pc].T].T
    # pc: vector from contact frame to wrench frame
    # [^pc]@R'
    # we can just get this using the adjoint matrix
    # since wrench_tf = wf_Ad_tf.T @ wrench_wf
    # and Jc = wf_Ad_tf.T[:, :3] # (..., 6, 3)

    cf_X_wf = transform_matrix_inverse(wf_X_cf)
    cf_Ad_wf = get_adjoint_matrix(T=cf_X_wf)  # (6x6) matrix (..., 6, 6)
    if isinstance(cf_Ad_wf, np.ndarray):
        # transpose the last two dimensions
        Jc = np.swapaxes(cf_Ad_wf, -1, -2)[..., :, :3] # (..., 6, 3)
    elif isinstance(cf_Ad_wf, torch.Tensor):
        Jc = cf_Ad_wf.swapdims(-1, -2)[..., :, :3] # (..., 6, 3)
    else:
        raise ValueError('cf_Ad_wf must be a numpy array or a torch tensor.')
    # cf_X_wf = transform_matrix_inverse(wf_X_cf)
    # wf_Ad_tf = get_adjoint_matrix(T=cf_X_wf)  # (6x6) matrix (..., 6, 6)
    # Jc = wf_Ad_tf.T[:, :3]  # (..., 6, 3)
    return Jc


def transform_twist(twist_f1:Union[list, np.ndarray, torch.Tensor], f2_X_f1:Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
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


def twist_to_twist_matrix(twist:Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert a twist to a matrix.
    :param twist: (..., 6) as [v_x, v_y, v_z, omega_x, omega_y, omega_z]
    :return: twist_matrix: (..., 4, 4) as the twist matrix
    twist_matrix is defined as:
        [  0, -wz,  wy, vx]
        [ wz,   0, -wx, vy]
        [-wy,  wx,   0, vz]
        [  0,   0,   0,  0]
    """
    if isinstance(twist, np.ndarray):
        twist_matrix = np.zeros(twist.shape[:-1] + (4, 4))
        twist_matrix[..., :3, :3] = vector_to_skew_matrix(twist[..., 3:])
        twist_matrix[..., :3, 3] = twist[..., :3]
    elif isinstance(twist, torch.Tensor):
        twist_matrix = torch.zeros(twist.shape[:-1] + (4, 4), dtype=twist.dtype, device=twist.device)
        twist_matrix[..., :3, :3] = vector_to_skew_matrix(twist[..., 3:])
        twist_matrix[..., :3, 3] = twist[..., :3]
    else:
        raise ValueError('twist must be a numpy array or a torch tensor.')
    return twist_matrix


def twist_matrix_to_twist(twist_matrix:Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert a twist matrix to a twist.
    :param twist_matrix: (..., 4, 4) as the twist matrix
    :return: twist: (..., 6) as [v_x, v_y, v_z, omega_x, omega_y, omega_z]
    """
    v = twist_matrix[..., :3, 3] # (..., 3)
    omega = skew_matrix_to_vector(twist_matrix[..., :3, :3]) # (..., 3)
    if isinstance(twist_matrix, np.ndarray):
        twist = np.concatenate([v, omega], axis=-1) # (..., 6)
    elif isinstance(twist_matrix, torch.Tensor):
        twist = torch.cat([v, omega], dim=-1) # (..., 6)
    else:
        raise ValueError('twist_matrix must be a numpy array or a torch tensor.')
    return twist


def twist_to_transform(twist:np.ndarray, dt:float=1.0) -> np.ndarray:
    """
    Apply the exponential map to a twist to get a transformation matrix.
    :param twist: (..., 6) as [v_x, v_y, v_z, omega_x, omega_y, omega_z]
    :param dt: (float) time step
    :return: exp_twist: (..., 4, 4) as the transformation matrix SE(3)
    NOTE: The exponential map of a twist is given by:
        exp: se(3) -> SE(3)  where  T = exp(twist_matrix*dt) where twist_matrix is the matrix of the twist
    """
    v = twist[..., :3] # (..., 3)
    omega = twist[..., 3:] # (..., 3)
    skew_matrix = vector_to_skew_matrix(omega) # (..., 3, 3)
    exp_skew_matrix = exponential_map(skew_matrix*dt) # (..., 3, 3)
    if isinstance(twist, np.ndarray):
        # compute the linear term
        omega_cross_v = np.einsum('...ij,...j->...i', skew_matrix, v) # (..., 3)
        omega_dot_v = np.einsum('...i,...i->...', omega, v) # (...,)
        l1 = np.einsum('...ij,...j->...i', np.eye(3) + exp_skew_matrix, omega_cross_v) # (..., 3)
        l2 = omega_dot_v * dt * omega # (..., 3)
        l = l1 + l2 # (..., 3)
        # form the matrix
        exp_twist = np.zeros(twist.shape[:-1] + (4, 4))
        exp_twist[..., :3, :3] = exp_skew_matrix
        exp_twist[..., :3, 3] = l
        exp_twist[..., 3, 3] = 1.0
    elif isinstance(twist, torch.Tensor):
        # compute the linear term
        omega_cross_v = torch.einsum('...ij,...j->...i', skew_matrix, v) # (..., 3)
        omega_dot_v = torch.einsum('...i,...i->...', omega, v) # (...,)
        l1 = torch.einsum('...ij,...j->...i', torch.eye(3, device=twist.device, dtype=twist.dtype) + exp_skew_matrix, omega_cross_v) # (..., 3)
        l2 = omega_dot_v * dt * omega # (..., 3)
        l = l1 + l2
        # form the matrix
        exp_twist = torch.zeros(twist.shape[:-1] + (4, 4), dtype=twist.dtype, device=twist.device)
        exp_twist[..., :3, :3] = exp_skew_matrix
        exp_twist[..., :3, 3] = l
        exp_twist[..., 3, 3] = 1.0
    else:
        raise ValueError('twist must be a numpy array or a torch tensor.')
    return exp_twist


def transform_wrench(wrench_wf:Union[list, np.ndarray, torch.Tensor], wf_T_tf:Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
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


def get_point_velocity_from_twist(twist_f1:Union[np.ndarray, torch.Tensor], points_f1:Union[np.ndarray, torch.Tensor], points_have_point_dimension:bool=True, only_points_are_batched:bool=False, only_twists_are_batched:bool=False) -> Union[np.ndarray, torch.Tensor]:
    """
    Get the velocity of a point given a twist.
    :param twist_f1: twist in frame 1 of shape (..., 6) or (..., K, 6) if only_twists_are_batched=True
    :param point_f1: point in frame 1 of shape (..., 3) or (..., K, 3) if points_have_point_dimension=True
    :return: vel_f1: velocity of the point in frame 1 (..., 3) or (..., K, 3) if points_have_point_dimension=True
    """
    twist_matrices_f1 = twist_to_twist_matrix(twist_f1) # (..., 4, 4)
    # NOTE: the point-matrix multiplication is the same as the one used in transform_points_3d.
    # therefore, we can use the same function to get the point velocity
    if points_have_point_dimension:
        if only_points_are_batched:
            einusm_key = 'ij,...kj->...ki'
        elif only_twists_are_batched:
            einusm_key = '...ij,kj->...ki'
        else:
            # both are batched
            einusm_key = '...ij,...kj->...ki'
    else:
        if only_points_are_batched:
            einusm_key = 'ij,...j->...i'
        elif only_twists_are_batched:
            einusm_key = '...ij,j->...i'
        else:
            # both are batched
            einusm_key = '...ij,...j->...i'
    if isinstance(points_f1, torch.Tensor):
        points_f1_hom = torch.cat(
            [points_f1, torch.ones_like(points_f1[..., :1], dtype=points_f1.dtype, device=points_f1.device)], dim=-1)
        point_velocity_f1_hom = torch.einsum(einusm_key, twist_matrices_f1, points_f1_hom)  # (..., K, 4)
    elif isinstance(points_f1, np.ndarray):
        points_f1_hom = np.concatenate([points_f1, np.ones_like(points_f1[..., :1])], axis=-1)
        point_velocity_f1_hom = np.einsum(einusm_key, twist_matrices_f1, points_f1_hom)  # (..., K, 4)
    else:
        raise ValueError('Unsupported type')
    point_velocity_f1 = point_velocity_f1_hom[..., :3]
    return point_velocity_f1



# ==================================================================================================
# ==================================================================================================
# AUXILIARY FUNCTIONS
# ==================================================================================================

def skew_matrix_array(v:np.ndarray) -> np.ndarray:
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


def skew_matrix_tensor(v:torch.Tensor) -> torch.Tensor:
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


def get_adjoint_matrix_numpy(R=None, t=None, T=None) -> np.ndarray:
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
    adjoint_matrix[..., :3, 3:] = np.einsum('...ij,...jk->...ik', vector_to_skew_matrix(t), R)
    return adjoint_matrix


def get_adjoint_matrix_torch(R=None, t=None, T=None) -> torch.Tensor:
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
    adjoint_matrix[..., :3, 3:] = torch.einsum('...ij,...jk->...ik', vector_to_skew_matrix(t), R)
    return adjoint_matrix

# ==================================================================================================





