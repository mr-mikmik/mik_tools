import numpy as np
import torch

from mik_tools import tr, pose_to_matrix, transform_matrix_inverse


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


def get_contact_jacobian(wf_X_cf):
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
        Jc = cf_Ad_wf.permute(-1, -2)[..., :, :3] # (..., 6, 3)
    else:
        raise ValueError('cf_Ad_wf must be a numpy array or a torch tensor.')
    # cf_X_wf = transform_matrix_inverse(wf_X_cf)
    # wf_Ad_tf = get_adjoint_matrix(T=cf_X_wf)  # (6x6) matrix (..., 6, 6)
    # Jc = wf_Ad_tf.T[:, :3]  # (..., 6, 3)
    return Jc


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

# ==================================================================================================





