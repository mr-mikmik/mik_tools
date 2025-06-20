from __future__ import division
import numpy as np
import torch
import warnings
import math
from typing import Union, List, Tuple

import torch.nn.functional as F
from mik_tools.tensor_utils import batched_eye_tensor, batched_eye_array
from mik_tools.mik_tf_tools.transformations.tr_tools import _AXES2TUPLE, _EPS, _NEXT_AXIS, _TUPLE2AXES, vector_norm, unit_vector, _sqrt_positive_part


def rotation_to_transform(rotation_matrix: Union[torch.tensor, np.ndarray]) -> Union[torch.tensor, np.ndarray]:
    """
    Extends SO(3) to SE(3) with zero translation
    Args:
        rotation_matrix (torch.tensor, np.ndarray): of shape (..., 3, 3)
    Returns:
        transform_matrix (torch.tensor, np.ndarray) of shape (..., 4, 4)
    """
    batch_size = rotation_matrix.shape[:-2]
    if torch.is_tensor(rotation_matrix):
        # torch case
        transform_matrix = torch.zeros(batch_size + (4, 4), dtype=rotation_matrix.dtype, device=rotation_matrix.device)
    else:
        # numpy case
        transform_matrix = np.zeros(batch_size + (4, 4))
    transform_matrix[..., :3, :3] = rotation_matrix
    transform_matrix[..., 3, 3] = 1
    return transform_matrix


def euler_matrix(ai:Union[torch.Tensor, np.ndarray, float, int], aj:Union[torch.Tensor, np.ndarray, float, int], ak:Union[torch.Tensor, np.ndarray, float, int], axes='sxyz') -> [torch.Tensor, np.ndarray]:
    """
    Return homogeneous rotation matrix from Euler angles and axis sequence.
    :param ai: (torch.Tensor or np.ndarray) of shape (...,)
    :param aj: (torch.Tensor or np.ndarray) of shape (...,)
    :param ak: (torch.Tensor or np.ndarray) of shape (...,)
    :param axes: str of axis sequence
    :return: (torch.Tensor or np.ndarray) of shape (..., 4, 4)
    """
    if torch.is_tensor(ai) and torch.is_tensor(aj) and torch.is_tensor(ak):
        M = euler_matrix_tensor(ai=ai, aj=aj, ak=ak, axes=axes)
    elif type(ai) is np.ndarray and type(aj) is np.ndarray and type(ak) is np.ndarray:
        M = euler_matrix_array(ai=ai, aj=aj, ak=ak, axes=axes)
    elif type(ai) in [float, int] and type(aj) in [float, int] and type(ak) in [float, int]:
        M = euler_matrix_array(ai=np.array([ai]), aj=np.array([aj]), ak=np.array([ak]), axes=axes)[0]
    else:
        raise NotImplementedError(f'Input types not tensor or array (types: ai {type(ai)} aj {type(aj)} ak {type(ak)}')
    return M


def euler_from_matrix(matrix:Union[torch.Tensor, np.ndarray], axes='sxyz') -> Union[Tuple[torch.Tensor,torch.Tensor,torch.Tensor], Tuple[np.ndarray,np.ndarray,np.ndarray]]:
    """
    Return Euler angles from rotation matrix for specified axis sequence.

    axes : One of 24 axis sequences as string or encoded tuple

    Note that many Euler angle triplets can describe one matrix.

    """
    if torch.is_tensor(matrix):
        ax, ay, az = euler_from_matrix_tensor(matrix=matrix, axes=axes)
    elif type(matrix) is np.ndarray:
        ax, ay, az = euler_from_matrix_array(matrix=matrix, axes=axes)
    else:
        raise NotImplementedError(f'Input types not tensor or array (types: matrix {type(matrix)}')
    return ax, ay, az


def euler_from_quaternion(quaternion, axes='sxyz'):
    """
    Return Euler angles from quaternion for specified axis sequence.
    """
    return euler_from_matrix(quaternion_matrix(quaternion), axes)


def quaternion_from_euler(ai, aj, ak, axes='sxyz') -> np.ndarray:
    """
    Return quaternion from Euler angles and axis sequence.
    ai, aj, ak : Euler's roll, pitch and yaw angles
        ai: float or numpy array of shape (...,)
        aj: float or numpy array of shape (...,)
        ak: float or numpy array of shape (...,)
    axes : One of 24 axis sequences as string or encoded tuple
    out:
        quaternion (np.ndarray) of shape (..., 4)
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai_half = ai * 0.5
    aj_half = aj * 0.5
    ak_half = ak * 0.5
    ci = np.cos(ai_half)
    si = np.sin(ai_half)
    cj = np.cos(aj_half)
    sj = np.sin(aj_half)
    ck = np.cos(ak_half)
    sk = np.sin(ak_half)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    batch_dims = () if type(ai) == float else ai.shape
    quaternion = np.empty(batch_dims + (4, ), dtype=np.float64)
    if repetition:
        quaternion[..., i] = cj*(cs + sc)
        quaternion[..., j] = sj*(cc + ss)
        quaternion[..., k] = sj*(cs - sc)
        quaternion[..., 3] = cj*(cc - ss)
    else:
        quaternion[..., i] = cj*sc - sj*cs
        quaternion[..., j] = cj*ss + sj*cc
        quaternion[..., k] = cj*cs - sj*sc
        quaternion[..., 3] = cj*cc + sj*ss
    if parity:
        quaternion[..., j] *= -1

    return quaternion


def quaternion_about_axis(angle:Union[torch.Tensor, np.ndarray], axis:Union[torch.Tensor, np.ndarray]):
    """
    Return quaternion for rotation about axis.
    Args:
        angle (np.ndarray or torch.Tensor): of shape (...)
        axis (np.ndarray or torch.Tensor): of shape (...,3)
    Returns:
        quaternion (np.ndarray or torch.Tensor): of shape (...,4)
    """
    if torch.is_tensor(axis):
        quaternion = quaternion_about_axis_tensor(angle=angle, axis=axis)
    elif type(axis) is np.ndarray:
        quaternion = quaternion_about_axis_array(angle=angle, axis=axis)
    elif type(axis) in [list, tuple]:
        quaternion = quaternion_about_axis_array(angle=np.array([angle]), axis=np.array([axis]))[0]
    else:
        raise NotImplementedError(f'Input types not tensor or array (types: axis {type(axis)} angle {type(angle)}')
    return quaternion


def quaternion_about_axis_array(angle:np.ndarray, axis:np.ndarray) -> np.ndarray:
    # angle: np.ndarray of shape (...,)
    # axis: np.ndarray of shape (...,3)
    axis_norm = np.linalg.norm(axis, axis=-1, keepdims=True) # (...,1)
    axis_unit = axis/axis_norm # (...,3) TODO: make this safe
    qxyz = axis_unit*np.expand_dims(np.sin(angle*0.5), axis=-1) # (..., 3)
    qw = np.expand_dims(np.cos(angle*0.5), axis=-1) # (...,1)
    quaternion = np.concatenate([qxyz, qw], axis=-1) # (..., 4)
    return quaternion


def quaternion_about_axis_tensor(angle:torch.Tensor, axis:torch.Tensor) -> torch.Tensor:
    # angle: torch.Tensor of shape (...,)
    # axis: torch.Tensor of shape (...,3)
    axis_norm = torch.linalg.norm(axis, axis=-1).unsqueeze(-1) # (..., 1)
    axis_unit = axis/axis_norm # (..., 3) TODO: make this safe
    qxyz = axis_unit*torch.sin(angle*0.5).unsqueeze(-1) # (..., 3)
    qw = torch.cos(angle*0.5).unsqueeze(-1)
    quaternion = torch.cat([qxyz, qw], dim=-1) # (..., 4)
    return quaternion


def axis_angle_from_quaternion(quaternion):
    """
    Return angle and axis from quaternion.
    """
    quaternion = np.array(quaternion[:4], dtype=np.float64, copy=True)
    qlen = vector_norm(quaternion)
    if qlen > _EPS:
        angle = 2.0*math.acos(quaternion[3])
        axis = quaternion[:3] / qlen
    else:
        angle = 0.0
        axis = np.array([1.0, 0.0, 0.0])
    if angle == 0.0:
        axis = np.array([1.0, 0.0, 0.0])
    axis = axis/vector_norm(axis) # make sure it is unit length
    return axis, angle


def axis_angle_to_quat(axis_angle: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Transform an axis_angle to quaternion
    Args:
        axis_angle (np.ndarray): of shape (..., 3) representing an orientation in the axis-angle format, where
         - angle is the axis_angle norm
         - axis is the direction of the vector

    Returns:
        quat (np.ndarray): of shape (..., 4)
    """
    if torch.is_tensor(axis_angle):
        quaternion = axis_angle_to_quat_tensor(axis_angle=axis_angle)
    elif type(axis_angle) is np.ndarray:
        quaternion = axis_angle_to_quat_array(axis_angle=axis_angle)
    else:
        raise NotImplementedError(f'Input types not tensor or array (types: axis_angle {type(axis_angle)}')
    return quaternion


def axis_angle_to_quat_array(axis_angle: np.ndarray) -> np.ndarray:
    """
    Args:
        axis_angle (np.ndarray): of shape (..., 3) representing an orientation in the axis-angle format, where
         - angle is the axis_angle norm
         - axis is the direction of the vector

    Returns:
        quat (np.ndarray): of shape (..., 4)
    """
    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True) # (..., 1)
    axis = axis_angle / angle # (..., 3)
    # edit axis so where angle is zero, the axis is [1, 0, 0]
    mask = angle < _EPS # (..., 1)
    axis[mask[0]] = np.array([1.0, 0.0, 0.0], dtype=axis.dtype)
    quat = quaternion_about_axis_array(angle=angle[...,0], axis=axis)
    return quat


def axis_angle_to_quat_tensor(axis_angle: torch.Tensor) -> torch.Tensor:
    """

    Args:
        axis_angle (torch.Tensor): of shape (..., 3) representing an orientation in the axis-angle format, where
         - angle is the axis_angle norm
         - axis is the direction of the vector

    Returns:
        quat (torch.Tensor): of shape (..., 4)
    """
    angle = torch.linalg.norm(axis_angle, dim=-1) # (...,)
    axis = axis_angle / angle.unsqueeze(-1) # (..., 3)
    # edit axis so where angle is zero, the axis is [1, 0, 0]
    mask = angle < _EPS
    axis[mask] = torch.tensor([1.0, 0.0, 0.0], dtype=axis.dtype, device=axis.device)
    quat = quaternion_about_axis_tensor(angle=angle, axis=axis)
    return quat


def quaternion_matrix(quaternion):
    """
    Return homogeneous rotation matrix from quaternion.
    """
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)


def quaternion_from_matrix(matrix):
    """
    Return quaternion from rotation matrix.
    """
    q = np.empty((4, ), dtype=np.float64)
    # M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    M = np.asarray(matrix)
    t = np.trace(M)
    if t > M[3, 3]:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    return q


def quaternion_matrix_batched_array(quaternion:np.ndarray) -> np.ndarray:
    """
    Convert rotations given as quaternions to rotation matrices.
    :param quaternion: np array of shape (..., 4)
    :return: matrix_out (..., 4, 4) as the rotation matrices
    """
    nq = np.einsum('...i,...i->...', quaternion, quaternion) # (...)
    quaternion = quaternion * np.sqrt(2.0 / nq[..., np.newaxis]) # (..., 4) # normalized quat
    outer_q = np.einsum('...i,...j->...ij', quaternion, quaternion)
    matrix_out = np.zeros(quaternion.shape[:-1] + (4, 4), dtype=quaternion.dtype)
    matrix_out[..., 0, 0] = 1.0 - outer_q[..., 1, 1] - outer_q[..., 2, 2]
    matrix_out[..., 0, 1] = outer_q[..., 0, 1] - outer_q[..., 2, 3]
    matrix_out[..., 0, 2] = outer_q[..., 0, 2] + outer_q[..., 1, 3]
    matrix_out[..., 1, 0] = outer_q[..., 0, 1] + outer_q[..., 2, 3]
    matrix_out[..., 1, 1] = 1.0 - outer_q[..., 0, 0] - outer_q[..., 2, 2]
    matrix_out[..., 1, 2] = outer_q[..., 1, 2] - outer_q[..., 0, 3]
    matrix_out[..., 2, 0] = outer_q[..., 0, 2] - outer_q[..., 1, 3]
    matrix_out[..., 2, 1] = outer_q[..., 1, 2] + outer_q[..., 0, 3]
    matrix_out[..., 2, 2] = 1.0 - outer_q[..., 0, 0] - outer_q[..., 1, 1]
    matrix_out[..., 3, 3] = 1.0
    # check if the norm squared is zero
    mask = nq < _EPS
    matrix_out[mask] = np.identity(4, dtype=quaternion.dtype)
    return matrix_out


def quaternion_matrix_batched_tensor(quaternion:torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.
    :param quaternion: torch tensor of shape (..., 4)
    :return: matrix_out (..., 4, 4) as the rotation matrices
    """
    nq = torch.einsum('...i,...i->...', quaternion, quaternion)  # (...)
    quaternion = quaternion * torch.sqrt(2.0 / nq.unsqueeze(-1))  # (..., 4) # normalized quat
    outer_q = torch.einsum('...i,...j->...ij', quaternion, quaternion)
    matrix_out = torch.zeros(quaternion.shape[:-1] + (4, 4), dtype=quaternion.dtype, device=quaternion.device)
    matrix_out[..., 0, 0] = 1.0 - outer_q[..., 1, 1] - outer_q[..., 2, 2]
    matrix_out[..., 0, 1] = outer_q[..., 0, 1] - outer_q[..., 2, 3]
    matrix_out[..., 0, 2] = outer_q[..., 0, 2] + outer_q[..., 1, 3]
    matrix_out[..., 1, 0] = outer_q[..., 0, 1] + outer_q[..., 2, 3]
    matrix_out[..., 1, 1] = 1.0 - outer_q[..., 0, 0] - outer_q[..., 2, 2]
    matrix_out[..., 1, 2] = outer_q[..., 1, 2] - outer_q[..., 0, 3]
    matrix_out[..., 2, 0] = outer_q[..., 0, 2] - outer_q[..., 1, 3]
    matrix_out[..., 2, 1] = outer_q[..., 1, 2] + outer_q[..., 0, 3]
    matrix_out[..., 2, 2] = 1.0 - outer_q[..., 0, 0] - outer_q[..., 1, 1]
    matrix_out[..., 3, 3] = 1.0
    # check if the norm squared is zero
    mask = nq < _EPS
    matrix_out[mask] = torch.eye(4, dtype=quaternion.dtype, device=quaternion.device)
    return matrix_out


def quaternion_from_matrix_batched_array(matrix):
    """

    :param matrix: np array of shape (..., 4, 4)
    :return: (..., 4)
    """
    # until we do not have a good implementation, we will just do a for loop
    in_shape = matrix.shape
    matrix = matrix.reshape(-1, 4, 4)
    quaternions_out = []
    for matrix_i in matrix:
        quat_i = quaternion_from_matrix(matrix_i)
        quaternions_out.append(quat_i)
    quaternions_out = np.stack(quaternions_out, axis=0)
    quaternions_out = quaternions_out.reshape(in_shape[:-2] + (4, ))
    return quaternions_out


def quaternion_from_matrix_batched_tensor(matrix_in: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix_in: Transformation matrices as tensor of shape (..., 4, 4).

    Returns:
        Quaternions with real part last, as tensor of shape (..., 4). Format: [qx, qy, qz, qw].
    """
    # Extract rotation part
    R = matrix_in[..., :3, :3]  # shape (..., 3, 3)

    batch_shape = R.shape[:-2]
    q = torch.zeros(batch_shape + (4,), dtype=R.dtype, device=R.device) # (..., 4)

    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] # (...,)

    # Positive trace case
    mask = trace > 0.0 # (...,)
    t = trace[mask] # (...,)
    s = torch.sqrt(t + 1.0) * 2  # s = 4 * qw # (...,)
    q[mask, 3] = 0.25 * s
    q[mask, 0] = (R[mask, 2, 1] - R[mask, 1, 2]) / s
    q[mask, 1] = (R[mask, 0, 2] - R[mask, 2, 0]) / s
    q[mask, 2] = (R[mask, 1, 0] - R[mask, 0, 1]) / s

    # Handle cases where trace <= 0
    mask1 = (R[..., 0, 0] > R[..., 1, 1]) & (R[..., 0, 0] > R[..., 2, 2]) & (~mask) # (...,)
    s1 = torch.sqrt(1.0 + R[mask1, 0, 0] - R[mask1, 1, 1] - R[mask1, 2, 2]) * 2
    q[mask1, 3] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s1
    q[mask1, 0] = 0.25 * s1
    q[mask1, 1] = (R[mask1, 0, 1] + R[mask1, 1, 0]) / s1
    q[mask1, 2] = (R[mask1, 0, 2] + R[mask1, 2, 0]) / s1

    mask2 = (R[..., 1, 1] > R[..., 2, 2]) & (~mask) & (~mask1) # (...,)
    s2 = torch.sqrt(1.0 + R[mask2, 1, 1] - R[mask2, 0, 0] - R[mask2, 2, 2]) * 2
    q[mask2, 3] = (R[mask2, 0, 2] - R[mask2, 2, 0]) / s2
    q[mask2, 0] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s2
    q[mask2, 1] = 0.25 * s2
    q[mask2, 2] = (R[mask2, 1, 2] + R[mask2, 2, 1]) / s2

    mask3 = (~mask) & (~mask1) & (~mask2) # (...,)
    s3 = torch.sqrt(1.0 + R[mask3, 2, 2] - R[mask3, 0, 0] - R[mask3, 1, 1]) * 2
    q[mask3, 3] = (R[mask3, 1, 0] - R[mask3, 0, 1]) / s3
    q[mask3, 0] = (R[mask3, 0, 2] + R[mask3, 2, 0]) / s3
    q[mask3, 1] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s3
    q[mask3, 2] = 0.25 * s3

    # Normalize quaternion to ensure unit length
    q = q / q.norm(dim=-1, keepdim=True) # (..., 4)

    return q




def quaternion_from_matrix_batched_tensor_2(matrix_in: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Transformation matrix matrices as tensor of shape (..., 4, 4).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4). as [qx, qy, qz, qw] i.e. [0, 0, 0, 1] is unit quat.
    """
    if matrix_in.size(-1) == 4 or matrix_in.size(-2) == 4:
        matrix = matrix_in[..., :3, :3]
    else:
        matrix = matrix_in
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    quaternions = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    # standardize the output
    quaternions_wxyz = torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)
    quaternions_xyzw = torch.cat([quaternions_wxyz[...,1:], quaternions[...,:1]], dim=-1)
    return quaternions_xyzw


def quaternion_multiply(quaternion1:Union[torch.Tensor, np.ndarray], quaternion0:Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Return multiplication of two quaternions.
    Args:
        quaternion1 (torch.Tensor or np.ndarray): of shape (..., 4) as (qx, qy, qz, qw)
        quaternion0 (torch.Tensor or np.ndarray): of shape (..., 4) as (qx, qy, qz, qw)
    Returns:
        q_out (torch.Tensor or np.ndarray): result of left multiplying q1*q0
    """
    x0, y0, z0, w0 = quaternion0[..., 0], quaternion0[..., 1], quaternion0[..., 2], quaternion0[..., 3]
    x1, y1, z1, w1 = quaternion1[..., 0], quaternion1[..., 1], quaternion1[..., 2], quaternion1[..., 3]
    q_mult = [
         x1*w0 + y1*z0 - z1*y0 + w1*x0,
        -x1*z0 + y1*w0 + z1*x0 + w1*y0,
         x1*y0 - y1*x0 + z1*w0 + w1*z0,
        -x1*x0 - y1*y0 - z1*z0 + w1*w0]
    if torch.is_tensor(quaternion1):
        q_out = torch.stack(q_mult, dim=-1)
    else:
        q_out = np.stack(q_mult, axis=-1)
    return q_out


def quaternion_conjugate(quaternion:Union[torch.Tensor, np.ndarray])->Union[torch.Tensor, np.ndarray]:
    """
    Return conjugate of quaternion.
    Args:
        quaternion (torch.Tensor or np.ndarray): of shape (..., 4) as (qx, qy, qz, qw)
    Returns:
        q_bar (torch.Tensor or np.ndarray): of shape (..., 4) as (qx, qy, qz, qw) -- conjugate of quaternion
    """
    qx, qy, qz, qw = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
    q_out = [-qx, -qy, -qz, qw]
    if torch.is_tensor(quaternion):
        q_out = torch.stack(q_out, dim=-1)
    else:
        q_out = np.stack(q_out, axis=-1)
    return q_out


def quaternion_inverse(quaternion:Union[torch.Tensor, np.ndarray])->Union[torch.Tensor, np.ndarray]:
    """
    Return inverse of quaternion.
    Args:
        quaternion (torch.Tensor or np.ndarray): of shape (..., 4) as (qx, qy, qz, qw)
    Returns:
        q_inv (torch.Tensor or np.ndarray): of shape (..., 4) as (qx, qy, qz, qw) -- inverse of quaternion
    """
    # q_dot = np.dot(quaternion, quaternion)
    if torch.is_tensor(quaternion):
        q_dot = torch.einsum('...i,...i->...', quaternion, quaternion).unsqueeze(-1) # (..., 1)
    else:
        q_dot = np.expand_dims(np.einsum('...i,...i->...', quaternion, quaternion), axis=-1) # (..., 1)
    q_conj = quaternion_conjugate(quaternion) # (..., 4)
    return q_conj / q_dot


def quaternion_slerp(quat0, quat1, fraction, spin=0, shortestpath=True):
    """
    Return spherical linear interpolation between two quaternions.
    """
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < _EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        q1 *= -1.0
    angle = math.acos(d) + spin * math.pi
    if abs(angle) < _EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0




def euler_matrix_raw(ai, aj, ak, axes='sxyz'):
    """
    Return homogeneous rotation matrix from Euler angles and axis sequence.
    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    M = np.identity(4)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj*si
        M[i, k] = sj*ci
        M[j, i] = sj*sk
        M[j, j] = -cj*ss+cc
        M[j, k] = -cj*cs-sc
        M[k, i] = -sj*ck
        M[k, j] = cj*sc+cs
        M[k, k] = cj*cc-ss
    else:
        M[i, i] = cj*ck
        M[i, j] = sj*sc-cs
        M[i, k] = sj*cc+ss
        M[j, i] = cj*sk
        M[j, j] = sj*ss+cc
        M[j, k] = sj*cs-sc
        M[k, i] = -sj
        M[k, j] = cj*si
        M[k, k] = cj*ci
    return M



def euler_matrix_array(ai:np.ndarray, aj:np.ndarray, ak:np.ndarray, axes='sxyz') -> np.ndarray:
    """
    Return homogeneous rotation matrix from Euler angles and axis sequence.
    :param ai: (...,) np array of roll angles
    :param aj: (...,) np array of pitch angles
    :param ak: (...,) np array of yaw angles
    :param axes: code for axis sequence as string or encoded tuple (default 'sxyz')
    :return matrix: (..., 4, 4) np array of rotation matrices
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak) # (...,), (...,), (...,)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak) # (...,), (...,), (...,)
    cc, cs = ci*ck, ci*sk # (...,), (...,)
    sc, ss = si*ck, si*sk # (...,), (...,)

    M = batched_eye_array(4, batch_shape=ai.shape) # (..., 4, 4)
    if repetition:
        M[..., i, i] = cj
        M[..., i, j] = sj*si
        M[..., i, k] = sj*ci
        M[..., j, i] = sj*sk
        M[..., j, j] = -cj*ss+cc
        M[..., j, k] = -cj*cs-sc
        M[..., k, i] = -sj*ck
        M[..., k, j] = cj*sc+cs
        M[..., k, k] = cj*cc-ss
    else:
        M[..., i, i] = cj*ck
        M[..., i, j] = sj*sc-cs
        M[..., i, k] = sj*cc+ss
        M[..., j, i] = cj*sk
        M[..., j, j] = sj*ss+cc
        M[..., j, k] = sj*cs-sc
        M[..., k, i] = -sj
        M[..., k, j] = cj*si
        M[..., k, k] = cj*ci
    return M


def euler_matrix_tensor(ai:torch.Tensor, aj:torch.Tensor, ak:torch.Tensor, axes='sxyz') -> torch.Tensor:
    """
    Return homogeneous rotation matrix from Euler angles and axis sequence.
    :param ai: (...,) torch tensor of roll angles
    :param aj: (...,) torch tensor of pitch angles
    :param ak: (...,) torch tensor of yaw angles
    :param axes: code for axis sequence as string or encoded tuple (default 'sxyz')
    :return matrix: (..., 4, 4) torch tensor of rotation matrices
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes] # (int, int, int, int)
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes # (int, int, int, int)

    i = firstaxis # (int)
    j = _NEXT_AXIS[i+parity] # (int)
    k = _NEXT_AXIS[i-parity+1] # (int)

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = torch.sin(ai), torch.sin(aj), torch.sin(ak) # (...,), (...,), (...,)
    ci, cj, ck = torch.cos(ai), torch.cos(aj), torch.cos(ak) # (...,), (...,), (...,)
    cc, cs = ci*ck, ci*sk # (...,), (...,)
    sc, ss = si*ck, si*sk # (...,), (...,)

    M = batched_eye_tensor(4, batch_shape=ai.shape, device=ai.device, dtype=ai.dtype) # (..., 4, 4)
    if repetition:
        M[..., i, i] = cj
        M[..., i, j] = sj*si
        M[..., i, k] = sj*ci
        M[..., j, i] = sj*sk
        M[..., j, j] = -cj*ss+cc
        M[..., j, k] = -cj*cs-sc
        M[..., k, i] = -sj*ck
        M[..., k, j] = cj*sc+cs
        M[..., k, k] = cj*cc-ss
    else:
        M[..., i, i] = cj*ck
        M[..., i, j] = sj*sc-cs
        M[..., i, k] = sj*cc+ss
        M[..., j, i] = cj*sk
        M[..., j, j] = sj*ss+cc
        M[..., j, k] = sj*cs-sc
        M[..., k, i] = -sj
        M[..., k, j] = cj*si
        M[..., k, k] = cj*ci
    return M


def euler_from_matrix_raw(matrix, axes='sxyz'):
    """
    Return Euler angles from rotation matrix for specified axis sequence.

    axes : One of 24 axis sequences as string or encoded tuple

    Note that many Euler angle triplets can describe one matrix.

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az


def euler_from_matrix_array(matrix:np.ndarray, axes='sxyz') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
     Return Euler angles from rotation matrix for specified axis sequence.

    axes :
    Note that many Euler angle triplets can describe one matrix.
    :param matrix (np.ndarray): of shape (..., 4, 4)
    :param axes (str): One of 24 axis sequences as string or encoded tuple
    :return ax (np.ndarray): of shape (...,)
    :return ay (np.ndarray): of shape (...,)
    :return az (np.ndarray): of shape (...,)
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = np.array(matrix, dtype=np.float64, copy=False)[..., :3, :3] # (..., 3, 3)
    if repetition:
        sy = np.sqrt(M[..., i, j]*M[..., i, j] + M[..., i, k]*M[..., i, k])
        if sy > _EPS:
            ax = np.arctan2( M[..., i, j],  M[..., i, k])
            ay = np.arctan2( sy,       M[..., i, i])
            az = np.arctan2( M[..., j, i], -M[..., k, i])
        else:
            ax = np.arctan2(-M[..., j, k],  M[..., j, j])
            ay = np.arctan2( sy,       M[..., i, i])
            az = 0.0 * sy
    else:
        cy = np.sqrt(M[..., i, i]*M[..., i, i] + M[..., j, i]*M[..., j, i])
        if cy > _EPS:
            ax = np.arctan2( M[..., k, j],  M[..., k, k])
            ay = np.arctan2(-M[..., k, i],  cy)
            az = np.arctan2( M[..., j, i],  M[..., i, i])
        else:
            ax = np.arctan2(-M[..., j, k],  M[..., j, j])
            ay = np.arctan2(-M[..., k, i],  cy)
            az = 0.0 * cy

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az


def euler_from_matrix_tensor(matrix:torch.Tensor, axes='sxyz') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
     Return Euler angles from rotation matrix for specified axis sequence.

    axes :
    Note that many Euler angle triplets can describe one matrix.
    :param matrix (torch.Tensor): of shape (..., 4, 4)
    :param axes (str): One of 24 axis sequences as string or encoded tuple
    :return ax (torch.Tensor): of shape (...,)
    :return ay (torch.Tensor): of shape (...,)
    :return az (torch.Tensor): of shape (...,)
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = matrix[..., :3, :3] # (..., 3, 3)
    if repetition:
        sy = torch.sqrt(M[..., i, j]*M[..., i, j] + M[..., i, k]*M[..., i, k])
        ay = torch.arctan2(sy, M[..., i, i])
        ax = torch.zeros_like(ay, dtype=ay.dtype, device=ay.device)
        az = torch.zeros_like(ay, dtype=ay.dtype, device=ay.device)
        mask = sy > _EPS
        ax[mask] = torch.arctan2( M[..., i, j][mask],  M[..., i, k][mask])
        az[mask] = torch.arctan2( M[..., j, i][mask], -M[..., k, i][mask])
        ax[~mask] = torch.arctan2(-M[..., j, k][~mask],  M[..., j, j][~mask])
    else:
        cy = torch.sqrt(M[..., i, i]*M[..., i, i] + M[..., j, i]*M[..., j, i])
        ay = torch.arctan2(-M[..., k, i], cy)
        ax = torch.zeros_like(ay, dtype=ay.dtype, device=ay.device)
        az = torch.zeros_like(ay, dtype=ay.dtype, device=ay.device)
        mask = cy > _EPS
        ax[mask] = torch.arctan2( M[..., k, j][mask],  M[..., k, k][mask])
        az[mask] = torch.arctan2( M[..., j, i][mask],  M[..., i, i][mask])
        ax[~mask] = torch.arctan2(-M[..., j, k][~mask],  M[..., j, j][~mask])

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az





