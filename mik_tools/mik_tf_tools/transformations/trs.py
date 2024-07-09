from __future__ import division
import numpy as np
import torch
import warnings
import math
from typing import Union

import torch.nn.functional as F
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









def euler_matrix(ai, aj, ak, axes='sxyz'):
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


def euler_from_matrix(matrix, axes='sxyz'):
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


def euler_from_quaternion(quaternion, axes='sxyz'):
    """
    Return Euler angles from quaternion for specified axis sequence.
    """
    return euler_from_matrix(quaternion_matrix(quaternion), axes)


def quaternion_from_euler(ai, aj, ak, axes='sxyz'):
    """
    Return quaternion from Euler angles and axis sequence.
    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple
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

    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    quaternion = np.empty((4, ), dtype=np.float64)
    if repetition:
        quaternion[i] = cj*(cs + sc)
        quaternion[j] = sj*(cc + ss)
        quaternion[k] = sj*(cs - sc)
        quaternion[3] = cj*(cc - ss)
    else:
        quaternion[i] = cj*sc - sj*cs
        quaternion[j] = cj*ss + sj*cc
        quaternion[k] = cj*cs - sj*sc
        quaternion[3] = cj*cc + sj*ss
    if parity:
        quaternion[j] *= -1

    return quaternion


def quaternion_about_axis(angle:Union[torch.Tensor, np.ndarray], axis:Union[torch.Tensor, np.ndarray]):
    """
    Return quaternion for rotation about axis.
    Args:
        angle (np.ndarray or torch.Tensor): of shape (...)
        axis ():

    Returns:

    >>> q = quaternion_about_axis(0.123, (1, 0, 0))
    >>> np.allclose(q, [0.06146124, 0, 0, 0.99810947])
    True

    """
    if torch.is_tensor(axis):
        quaternion = quaternion_about_axis_tensor(angle=angle, axis=axis)
    elif type(axis) is np.ndarray:
        quaternion = quaternion_about_axis_array(angle=angle, axis=axis)
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
    axis = axis_angle / angle
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
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
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


def quaternion_matrix_batched_array(quaternion):
    """

    :param quaternion: np array of shape (..., 4)
    :return: (..., 4, 4)
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


def quaternion_matrix_batched_tensor(quaternion):
    """

    :param quaternion: torch tensor of shape (..., 4)
    :return: (..., 4, 4)
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
        matrix: Transformation matrix matrices as tensor of shape (..., 4, 4).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
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
    quaternions = torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)
    return quaternions


def quaternion_multiply(quaternion1, quaternion0):
    """
    Return multiplication of two quaternions.
    """
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array((
         x1*w0 + y1*z0 - z1*y0 + w1*x0,
        -x1*z0 + y1*w0 + z1*x0 + w1*y0,
         x1*y0 - y1*x0 + z1*w0 + w1*z0,
        -x1*x0 - y1*y0 - z1*z0 + w1*w0), dtype=np.float64)


def quaternion_conjugate(quaternion):
    """
    Return conjugate of quaternion.
    """
    return np.array((-quaternion[0], -quaternion[1],
                        -quaternion[2], quaternion[3]), dtype=np.float64)


def quaternion_inverse(quaternion):
    """
    Return inverse of quaternion.
    """
    return quaternion_conjugate(quaternion) / np.dot(quaternion, quaternion)


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


