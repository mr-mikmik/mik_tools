# Code addapted from Christoph Gohlke Homogeneous Transformation Matrices and Quaternions.
from __future__ import division
import numpy as np
import torch
import math
import torch.nn.functional as F
import warnings
from mik_tools.mik_tf_tools.transformations.tr_tools import _EPS, unit_vector, vector_norm, is_same_transform


def identity_matrix():
    """Return 4x4 identity/unit matrix.

    >>> I = identity_matrix()
    >>> np.allclose(I, np.dot(I, I))
    True
    >>> np.sum(I), np.trace(I)
    (4.0, 4.0)
    >>> np.allclose(I, np.identity(4, dtype=np.float64))
    True

    """
    return np.identity(4, dtype=np.float64)


def translation_matrix(direction):
    """Return matrix to translate by direction vector.

    >>> v = np.random.random(3) - 0.5
    >>> np.allclose(v, translation_matrix(v)[:3, 3])
    True

    """
    M = np.identity(4)
    M[:3, 3] = direction[:3]
    return M


def translation_from_matrix(matrix):
    """Return translation vector from translation matrix.

    >>> v0 = np.random.random(3) - 0.5
    >>> v1 = translation_from_matrix(translation_matrix(v0))
    >>> np.allclose(v0, v1)
    True

    """
    return np.array(matrix, copy=False)[:3, 3].copy()


def reflection_matrix(point, normal):
    """Return matrix to mirror at plane defined by point and normal vector.

    >>> v0 = np.random.random(4) - 0.5
    >>> v0[3] = 1.0
    >>> v1 = np.random.random(3) - 0.5
    >>> R = reflection_matrix(v0, v1)
    >>> np.allclose(2., np.trace(R))
    True
    >>> np.allclose(v0, np.dot(R, v0))
    True
    >>> v2 = v0.copy()
    >>> v2[:3] += v1
    >>> v3 = v0.copy()
    >>> v2[:3] -= v1
    >>> np.allclose(v2, np.dot(R, v3))
    True

    """
    normal = unit_vector(normal[:3])
    M = np.identity(4)
    M[:3, :3] -= 2.0 * np.outer(normal, normal)
    M[:3, 3] = (2.0 * np.dot(point[:3], normal)) * normal
    return M


def reflection_from_matrix(matrix):
    """Return mirror plane point and normal vector from reflection matrix.

    >>> v0 = np.random.random(3) - 0.5
    >>> v1 = np.random.random(3) - 0.5
    >>> M0 = reflection_matrix(v0, v1)
    >>> point, normal = reflection_from_matrix(M0)
    >>> M1 = reflection_matrix(point, normal)
    >>> is_same_transform(M0, M1)
    True

    """
    M = np.array(matrix, dtype=np.float64, copy=False)
    # normal: unit eigenvector corresponding to eigenvalue -1
    l, V = np.linalg.eig(M[:3, :3])
    i = np.where(abs(np.real(l) + 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue -1")
    normal = np.real(V[:, i[0]]).squeeze()
    # point: any unit eigenvector corresponding to eigenvalue 1
    l, V = np.linalg.eig(M)
    i = np.where(abs(np.real(l) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    point = np.real(V[:, i[-1]]).squeeze()
    point /= point[3]
    return point, normal


def rotation_matrix(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and direction.

    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = np.random.random(3) - 0.5
    >>> point = np.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(-angle, -direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> I = np.identity(4, np.float64)
    >>> np.allclose(I, rotation_matrix(math.pi*2, direc))
    True
    >>> np.allclose(2., np.trace(rotation_matrix(math.pi/2,
    ...                                                direc, point)))
    True

    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.array(((cosa, 0.0,  0.0),
                     (0.0,  cosa, 0.0),
                     (0.0,  0.0,  cosa)), dtype=np.float64)
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array((( 0.0,         -direction[2],  direction[1]),
                      ( direction[2], 0.0,          -direction[0]),
                      (-direction[1], direction[0],  0.0)),
                     dtype=np.float64)
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


def rotation_from_matrix(matrix):
    """Return rotation angle and axis from rotation matrix.

    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = np.random.random(3) - 0.5
    >>> point = np.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> angle, direc, point = rotation_from_matrix(R0)
    >>> R1 = rotation_matrix(angle, direc, point)
    >>> is_same_transform(R0, R1)
    True

    """
    R = np.array(matrix, dtype=np.float64, copy=False)
    R33 = R[:3, :3]
    # direction: unit eigenvector of R33 corresponding to eigenvalue of 1
    l, W = np.linalg.eig(R33.T)
    i = np.where(abs(np.real(l) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    direction = np.real(W[:, i[-1]]).squeeze()
    # point: unit eigenvector of R33 corresponding to eigenvalue of 1
    l, Q = np.linalg.eig(R)
    i = np.where(abs(np.real(l) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    point = np.real(Q[:, i[-1]]).squeeze()
    point /= point[3]
    # rotation angle depending on direction
    cosa = (np.trace(R33) - 1.0) / 2.0
    if abs(direction[2]) > 1e-8:
        sina = (R[1, 0] + (cosa-1.0)*direction[0]*direction[1]) / direction[2]
    elif abs(direction[1]) > 1e-8:
        sina = (R[0, 2] + (cosa-1.0)*direction[0]*direction[2]) / direction[1]
    else:
        sina = (R[2, 1] + (cosa-1.0)*direction[1]*direction[2]) / direction[0]
    angle = math.atan2(sina, cosa)
    return angle, direction, point


def scale_matrix(factor, origin=None, direction=None):
    """Return matrix to scale by factor around origin in direction.

    Use factor -1 for point symmetry.

    >>> v = (np.random.rand(4, 5) - 0.5) * 20.0
    >>> v[3] = 1.0
    >>> S = scale_matrix(-1.234)
    >>> np.allclose(np.dot(S, v)[:3], -1.234*v[:3])
    True
    >>> factor = random.random() * 10 - 5
    >>> origin = np.random.random(3) - 0.5
    >>> direct = np.random.random(3) - 0.5
    >>> S = scale_matrix(factor, origin)
    >>> S = scale_matrix(factor, origin, direct)

    """
    if direction is None:
        # uniform scaling
        M = np.array(((factor, 0.0,    0.0,    0.0),
                         (0.0,    factor, 0.0,    0.0),
                         (0.0,    0.0,    factor, 0.0),
                         (0.0,    0.0,    0.0,    1.0)), dtype=np.float64)
        if origin is not None:
            M[:3, 3] = origin[:3]
            M[:3, 3] *= 1.0 - factor
    else:
        # nonuniform scaling
        direction = unit_vector(direction[:3])
        factor = 1.0 - factor
        M = np.identity(4)
        M[:3, :3] -= factor * np.outer(direction, direction)
        if origin is not None:
            M[:3, 3] = (factor * np.dot(origin[:3], direction)) * direction
    return M


def scale_from_matrix(matrix):
    """Return scaling factor, origin and direction from scaling matrix.

    >>> factor = random.random() * 10 - 5
    >>> origin = np.random.random(3) - 0.5
    >>> direct = np.random.random(3) - 0.5
    >>> S0 = scale_matrix(factor, origin)
    >>> factor, origin, direction = scale_from_matrix(S0)
    >>> S1 = scale_matrix(factor, origin, direction)
    >>> is_same_transform(S0, S1)
    True
    >>> S0 = scale_matrix(factor, origin, direct)
    >>> factor, origin, direction = scale_from_matrix(S0)
    >>> S1 = scale_matrix(factor, origin, direction)
    >>> is_same_transform(S0, S1)
    True

    """
    M = np.array(matrix, dtype=np.float64, copy=False)
    M33 = M[:3, :3]
    factor = np.trace(M33) - 2.0
    try:
        # direction: unit eigenvector corresponding to eigenvalue factor
        l, V = np.linalg.eig(M33)
        i = np.where(abs(np.real(l) - factor) < 1e-8)[0][0]
        direction = np.real(V[:, i]).squeeze()
        direction /= vector_norm(direction)
    except IndexError:
        # uniform scaling
        factor = (factor + 2.0) / 3.0
        direction = None
    # origin: any eigenvector corresponding to eigenvalue 1
    l, V = np.linalg.eig(M)
    i = np.where(abs(np.real(l) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no eigenvector corresponding to eigenvalue 1")
    origin = np.real(V[:, i[-1]]).squeeze()
    origin /= origin[3]
    return factor, origin, direction


def projection_matrix(point, normal, direction=None,
                      perspective=None, pseudo=False):
    """Return matrix to project onto plane defined by point and normal.

    Using either perspective point, projection direction, or none of both.

    If pseudo is True, perspective projections will preserve relative depth
    such that Perspective = dot(Orthogonal, PseudoPerspective).

    >>> P = projection_matrix((0, 0, 0), (1, 0, 0))
    >>> np.allclose(P[1:, 1:], np.identity(4)[1:, 1:])
    True
    >>> point = np.random.random(3) - 0.5
    >>> normal = np.random.random(3) - 0.5
    >>> direct = np.random.random(3) - 0.5
    >>> persp = np.random.random(3) - 0.5
    >>> P0 = projection_matrix(point, normal)
    >>> P1 = projection_matrix(point, normal, direction=direct)
    >>> P2 = projection_matrix(point, normal, perspective=persp)
    >>> P3 = projection_matrix(point, normal, perspective=persp, pseudo=True)
    >>> is_same_transform(P2, np.dot(P0, P3))
    True
    >>> P = projection_matrix((3, 0, 0), (1, 1, 0), (1, 0, 0))
    >>> v0 = (np.random.rand(4, 5) - 0.5) * 20.0
    >>> v0[3] = 1.0
    >>> v1 = np.dot(P, v0)
    >>> np.allclose(v1[1], v0[1])
    True
    >>> np.allclose(v1[0], 3.0-v1[1])
    True

    """
    M = np.identity(4)
    point = np.array(point[:3], dtype=np.float64, copy=False)
    normal = unit_vector(normal[:3])
    if perspective is not None:
        # perspective projection
        perspective = np.array(perspective[:3], dtype=np.float64,
                                  copy=False)
        M[0, 0] = M[1, 1] = M[2, 2] = np.dot(perspective-point, normal)
        M[:3, :3] -= np.outer(perspective, normal)
        if pseudo:
            # preserve relative depth
            M[:3, :3] -= np.outer(normal, normal)
            M[:3, 3] = np.dot(point, normal) * (perspective+normal)
        else:
            M[:3, 3] = np.dot(point, normal) * perspective
        M[3, :3] = -normal
        M[3, 3] = np.dot(perspective, normal)
    elif direction is not None:
        # parallel projection
        direction = np.array(direction[:3], dtype=np.float64, copy=False)
        scale = np.dot(direction, normal)
        M[:3, :3] -= np.outer(direction, normal) / scale
        M[:3, 3] = direction * (np.dot(point, normal) / scale)
    else:
        # orthogonal projection
        M[:3, :3] -= np.outer(normal, normal)
        M[:3, 3] = np.dot(point, normal) * normal
    return M


def projection_from_matrix(matrix, pseudo=False):
    """Return projection plane and perspective point from projection matrix.

    Return values are same as arguments for projection_matrix function:
    point, normal, direction, perspective, and pseudo.

    >>> point = np.random.random(3) - 0.5
    >>> normal = np.random.random(3) - 0.5
    >>> direct = np.random.random(3) - 0.5
    >>> persp = np.random.random(3) - 0.5
    >>> P0 = projection_matrix(point, normal)
    >>> result = projection_from_matrix(P0)
    >>> P1 = projection_matrix(*result)
    >>> is_same_transform(P0, P1)
    True
    >>> P0 = projection_matrix(point, normal, direct)
    >>> result = projection_from_matrix(P0)
    >>> P1 = projection_matrix(*result)
    >>> is_same_transform(P0, P1)
    True
    >>> P0 = projection_matrix(point, normal, perspective=persp, pseudo=False)
    >>> result = projection_from_matrix(P0, pseudo=False)
    >>> P1 = projection_matrix(*result)
    >>> is_same_transform(P0, P1)
    True
    >>> P0 = projection_matrix(point, normal, perspective=persp, pseudo=True)
    >>> result = projection_from_matrix(P0, pseudo=True)
    >>> P1 = projection_matrix(*result)
    >>> is_same_transform(P0, P1)
    True

    """
    M = np.array(matrix, dtype=np.float64, copy=False)
    M33 = M[:3, :3]
    l, V = np.linalg.eig(M)
    i = np.where(abs(np.real(l) - 1.0) < 1e-8)[0]
    if not pseudo and len(i):
        # point: any eigenvector corresponding to eigenvalue 1
        point = np.real(V[:, i[-1]]).squeeze()
        point /= point[3]
        # direction: unit eigenvector corresponding to eigenvalue 0
        l, V = np.linalg.eig(M33)
        i = np.where(abs(np.real(l)) < 1e-8)[0]
        if not len(i):
            raise ValueError("no eigenvector corresponding to eigenvalue 0")
        direction = np.real(V[:, i[0]]).squeeze()
        direction /= vector_norm(direction)
        # normal: unit eigenvector of M33.T corresponding to eigenvalue 0
        l, V = np.linalg.eig(M33.T)
        i = np.where(abs(np.real(l)) < 1e-8)[0]
        if len(i):
            # parallel projection
            normal = np.real(V[:, i[0]]).squeeze()
            normal /= vector_norm(normal)
            return point, normal, direction, None, False
        else:
            # orthogonal projection, where normal equals direction vector
            return point, direction, None, None, False
    else:
        # perspective projection
        i = np.where(abs(np.real(l)) > 1e-8)[0]
        if not len(i):
            raise ValueError(
                "no eigenvector not corresponding to eigenvalue 0")
        point = np.real(V[:, i[-1]]).squeeze()
        point /= point[3]
        normal = - M[3, :3]
        perspective = M[:3, 3] / np.dot(point[:3], normal)
        if pseudo:
            perspective -= normal
        return point, normal, None, perspective, pseudo


def clip_matrix(left, right, bottom, top, near, far, perspective=False):
    """Return matrix to obtain normalized device coordinates from frustrum.

    The frustrum bounds are axis-aligned along x (left, right),
    y (bottom, top) and z (near, far).

    Normalized device coordinates are in range [-1, 1] if coordinates are
    inside the frustrum.

    If perspective is True the frustrum is a truncated pyramid with the
    perspective point at origin and direction along z axis, otherwise an
    orthographic canonical view volume (a box).

    Homogeneous coordinates transformed by the perspective clip matrix
    need to be dehomogenized (devided by w coordinate).

    >>> frustrum = np.random.rand(6)
    >>> frustrum[1] += frustrum[0]
    >>> frustrum[3] += frustrum[2]
    >>> frustrum[5] += frustrum[4]
    >>> M = clip_matrix(*frustrum, perspective=False)
    >>> np.dot(M, [frustrum[0], frustrum[2], frustrum[4], 1.0])
    array([-1., -1., -1.,  1.])
    >>> np.dot(M, [frustrum[1], frustrum[3], frustrum[5], 1.0])
    array([ 1.,  1.,  1.,  1.])
    >>> M = clip_matrix(*frustrum, perspective=True)
    >>> v = np.dot(M, [frustrum[0], frustrum[2], frustrum[4], 1.0])
    >>> v / v[3]
    array([-1., -1., -1.,  1.])
    >>> v = np.dot(M, [frustrum[1], frustrum[3], frustrum[4], 1.0])
    >>> v / v[3]
    array([ 1.,  1., -1.,  1.])

    """
    if left >= right or bottom >= top or near >= far:
        raise ValueError("invalid frustrum")
    if perspective:
        if near <= _EPS:
            raise ValueError("invalid frustrum: near <= 0")
        t = 2.0 * near
        M = ((-t/(right-left), 0.0, (right+left)/(right-left), 0.0),
             (0.0, -t/(top-bottom), (top+bottom)/(top-bottom), 0.0),
             (0.0, 0.0, -(far+near)/(far-near), t*far/(far-near)),
             (0.0, 0.0, -1.0, 0.0))
    else:
        M = ((2.0/(right-left), 0.0, 0.0, (right+left)/(left-right)),
             (0.0, 2.0/(top-bottom), 0.0, (top+bottom)/(bottom-top)),
             (0.0, 0.0, 2.0/(far-near), (far+near)/(near-far)),
             (0.0, 0.0, 0.0, 1.0))
    return np.array(M, dtype=np.float64)


def shear_matrix(angle, direction, point, normal):
    """Return matrix to shear by angle along direction vector on shear plane.

    The shear plane is defined by a point and normal vector. The direction
    vector must be orthogonal to the plane's normal vector.

    A point P is transformed by the shear matrix into P" such that
    the vector P-P" is parallel to the direction vector and its extent is
    given by the angle of P-P'-P", where P' is the orthogonal projection
    of P onto the shear plane.

    >>> angle = (random.random() - 0.5) * 4*math.pi
    >>> direct = np.random.random(3) - 0.5
    >>> point = np.random.random(3) - 0.5
    >>> normal = np.cross(direct, np.random.random(3))
    >>> S = shear_matrix(angle, direct, point, normal)
    >>> np.allclose(1.0, np.linalg.det(S))
    True

    """
    normal = unit_vector(normal[:3])
    direction = unit_vector(direction[:3])
    if abs(np.dot(normal, direction)) > 1e-6:
        raise ValueError("direction and normal vectors are not orthogonal")
    angle = math.tan(angle)
    M = np.identity(4)
    M[:3, :3] += angle * np.outer(direction, normal)
    M[:3, 3] = -angle * np.dot(point[:3], normal) * direction
    return M


def shear_from_matrix(matrix):
    """Return shear angle, direction and plane from shear matrix.

    >>> angle = (random.random() - 0.5) * 4*math.pi
    >>> direct = np.random.random(3) - 0.5
    >>> point = np.random.random(3) - 0.5
    >>> normal = np.cross(direct, np.random.random(3))
    >>> S0 = shear_matrix(angle, direct, point, normal)
    >>> angle, direct, point, normal = shear_from_matrix(S0)
    >>> S1 = shear_matrix(angle, direct, point, normal)
    >>> is_same_transform(S0, S1)
    True

    """
    M = np.array(matrix, dtype=np.float64, copy=False)
    M33 = M[:3, :3]
    # normal: cross independent eigenvectors corresponding to the eigenvalue 1
    l, V = np.linalg.eig(M33)
    i = np.where(abs(np.real(l) - 1.0) < 1e-4)[0]
    if len(i) < 2:
        raise ValueError("No two linear independent eigenvectors found {}".format(l))
    V = np.real(V[:, i]).squeeze().T
    lenorm = -1.0
    for i0, i1 in ((0, 1), (0, 2), (1, 2)):
        n = np.cross(V[i0], V[i1])
        l = vector_norm(n)
        if l > lenorm:
            lenorm = l
            normal = n
    normal /= lenorm
    # direction and angle
    direction = np.dot(M33 - np.identity(3), normal)
    angle = vector_norm(direction)
    direction /= angle
    angle = math.atan(angle)
    # point: eigenvector corresponding to eigenvalue 1
    l, V = np.linalg.eig(M)
    i = np.where(abs(np.real(l) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no eigenvector corresponding to eigenvalue 1")
    point = np.real(V[:, i[-1]]).squeeze()
    point /= point[3]
    return angle, direction, point, normal


def decompose_matrix(matrix):
    """Return sequence of transformations from transformation matrix.

    matrix : array_like
        Non-degenerative homogeneous transformation matrix

    Return tuple of:
        scale : vector of 3 scaling factors
        shear : list of shear factors for x-y, x-z, y-z axes
        angles : list of Euler angles about static x, y, z axes
        translate : translation vector along x, y, z axes
        perspective : perspective partition of matrix

    Raise ValueError if matrix is of wrong type or degenerative.

    >>> T0 = translation_matrix((1, 2, 3))
    >>> scale, shear, angles, trans, persp = decompose_matrix(T0)
    >>> T1 = translation_matrix(trans)
    >>> np.allclose(T0, T1)
    True
    >>> S = scale_matrix(0.123)
    >>> scale, shear, angles, trans, persp = decompose_matrix(S)
    >>> scale[0]
    0.123
    >>> R0 = euler_matrix(1, 2, 3)
    >>> scale, shear, angles, trans, persp = decompose_matrix(R0)
    >>> R1 = euler_matrix(*angles)
    >>> np.allclose(R0, R1)
    True

    """
    M = np.array(matrix, dtype=np.float64, copy=True).T
    if abs(M[3, 3]) < _EPS:
        raise ValueError("M[3, 3] is zero")
    M /= M[3, 3]
    P = M.copy()
    P[:, 3] = 0, 0, 0, 1
    if not np.linalg.det(P):
        raise ValueError("Matrix is singular")

    scale = np.zeros((3, ), dtype=np.float64)
    shear = [0, 0, 0]
    angles = [0, 0, 0]

    if any(abs(M[:3, 3]) > _EPS):
        perspective = np.dot(M[:, 3], np.linalg.inv(P.T))
        M[:, 3] = 0, 0, 0, 1
    else:
        perspective = np.array((0, 0, 0, 1), dtype=np.float64)

    translate = M[3, :3].copy()
    M[3, :3] = 0

    row = M[:3, :3].copy()
    scale[0] = vector_norm(row[0])
    row[0] /= scale[0]
    shear[0] = np.dot(row[0], row[1])
    row[1] -= row[0] * shear[0]
    scale[1] = vector_norm(row[1])
    row[1] /= scale[1]
    shear[0] /= scale[1]
    shear[1] = np.dot(row[0], row[2])
    row[2] -= row[0] * shear[1]
    shear[2] = np.dot(row[1], row[2])
    row[2] -= row[1] * shear[2]
    scale[2] = vector_norm(row[2])
    row[2] /= scale[2]
    shear[1:] /= scale[2]

    if np.dot(row[0], np.cross(row[1], row[2])) < 0:
        scale *= -1
        row *= -1

    angles[1] = math.asin(-row[0, 2])
    if math.cos(angles[1]):
        angles[0] = math.atan2(row[1, 2], row[2, 2])
        angles[2] = math.atan2(row[0, 1], row[0, 0])
    else:
        #angles[0] = math.atan2(row[1, 0], row[1, 1])
        angles[0] = math.atan2(-row[2, 1], row[1, 1])
        angles[2] = 0.0

    return scale, shear, angles, translate, perspective


# def compose_matrix(scale=None, shear=None, angles=None, translate=None,
#                    perspective=None):
#     """Return transformation matrix from sequence of transformations.
# 
#     This is the inverse of the decompose_matrix function.
# 
#     Sequence of transformations:
#         scale : vector of 3 scaling factors
#         shear : list of shear factors for x-y, x-z, y-z axes
#         angles : list of Euler angles about static x, y, z axes
#         translate : translation vector along x, y, z axes
#         perspective : perspective partition of matrix
# 
#     >>> scale = np.random.random(3) - 0.5
#     >>> shear = np.random.random(3) - 0.5
#     >>> angles = (np.random.random(3) - 0.5) * (2*math.pi)
#     >>> trans = np.random.random(3) - 0.5
#     >>> persp = np.random.random(4) - 0.5
#     >>> M0 = compose_matrix(scale, shear, angles, trans, persp)
#     >>> result = decompose_matrix(M0)
#     >>> M1 = compose_matrix(*result)
#     >>> is_same_transform(M0, M1)
#     True
# 
#     """
#     M = np.identity(4)
#     if perspective is not None:
#         P = np.identity(4)
#         P[3, :] = perspective[:4]
#         M = np.dot(M, P)
#     if translate is not None:
#         T = np.identity(4)
#         T[:3, 3] = translate[:3]
#         M = np.dot(M, T)
#     if angles is not None:
#         R = euler_matrix(angles[0], angles[1], angles[2], 'sxyz')
#         M = np.dot(M, R)
#     if shear is not None:
#         Z = np.identity(4)
#         Z[1, 2] = shear[2]
#         Z[0, 2] = shear[1]
#         Z[0, 1] = shear[0]
#         M = np.dot(M, Z)
#     if scale is not None:
#         S = np.identity(4)
#         S[0, 0] = scale[0]
#         S[1, 1] = scale[1]
#         S[2, 2] = scale[2]
#         M = np.dot(M, S)
#     M /= M[3, 3]
#     return M


def orthogonalization_matrix(lengths, angles):
    """Return orthogonalization matrix for crystallographic cell coordinates.

    Angles are expected in degrees.

    The de-orthogonalization matrix is the inverse.

    >>> O = orthogonalization_matrix((10., 10., 10.), (90., 90., 90.))
    >>> np.allclose(O[:3, :3], np.identity(3, float) * 10)
    True
    >>> O = orthogonalization_matrix([9.8, 12.0, 15.5], [87.2, 80.7, 69.7])
    >>> np.allclose(np.sum(O), 43.063229)
    True

    """
    a, b, c = lengths
    angles = np.radians(angles)
    sina, sinb, _ = np.sin(angles)
    cosa, cosb, cosg = np.cos(angles)
    co = (cosa * cosb - cosg) / (sina * sinb)
    return np.array((
        ( a*sinb*math.sqrt(1.0-co*co),  0.0,    0.0, 0.0),
        (-a*sinb*co,                    b*sina, 0.0, 0.0),
        ( a*cosb,                       b*cosa, c,   0.0),
        ( 0.0,                          0.0,    0.0, 1.0)),
        dtype=np.float64)

# 
# def superimposition_matrix(v0, v1, scaling=False, usesvd=True):
#     """Return matrix to transform given vector set into second vector set.
# 
#     v0 and v1 are shape (3, \*) or (4, \*) arrays of at least 3 vectors.
# 
#     If usesvd is True, the weighted sum of squared deviations (RMSD) is
#     minimized according to the algorithm by W. Kabsch [8]. Otherwise the
#     quaternion based algorithm by B. Horn [9] is used (slower when using
#     this Python implementation).
# 
#     The returned matrix performs rotation, translation and uniform scaling
#     (if specified).
# 
#     >>> v0 = np.random.rand(3, 10)
#     >>> M = superimposition_matrix(v0, v0)
#     >>> np.allclose(M, np.identity(4))
#     True
#     >>> R = random_rotation_matrix(np.random.random(3))
#     >>> v0 = ((1,0,0), (0,1,0), (0,0,1), (1,1,1))
#     >>> v1 = np.dot(R, v0)
#     >>> M = superimposition_matrix(v0, v1)
#     >>> np.allclose(v1, np.dot(M, v0))
#     True
#     >>> v0 = (np.random.rand(4, 100) - 0.5) * 20.0
#     >>> v0[3] = 1.0
#     >>> v1 = np.dot(R, v0)
#     >>> M = superimposition_matrix(v0, v1)
#     >>> np.allclose(v1, np.dot(M, v0))
#     True
#     >>> S = scale_matrix(random.random())
#     >>> T = translation_matrix(np.random.random(3)-0.5)
#     >>> M = concatenate_matrices(T, R, S)
#     >>> v1 = np.dot(M, v0)
#     >>> v0[:3] += np.random.normal(0.0, 1e-9, 300).reshape(3, -1)
#     >>> M = superimposition_matrix(v0, v1, scaling=True)
#     >>> np.allclose(v1, np.dot(M, v0))
#     True
#     >>> M = superimposition_matrix(v0, v1, scaling=True, usesvd=False)
#     >>> np.allclose(v1, np.dot(M, v0))
#     True
#     >>> v = np.empty((4, 100, 3), dtype=np.float64)
#     >>> v[:, :, 0] = v0
#     >>> M = superimposition_matrix(v0, v1, scaling=True, usesvd=False)
#     >>> np.allclose(v1, np.dot(M, v[:, :, 0]))
#     True
# 
#     """
#     v0 = np.array(v0, dtype=np.float64, copy=False)[:3]
#     v1 = np.array(v1, dtype=np.float64, copy=False)[:3]
# 
#     if v0.shape != v1.shape or v0.shape[1] < 3:
#         raise ValueError("Vector sets are of wrong shape or type.")
# 
#     # move centroids to origin
#     t0 = np.mean(v0, axis=1)
#     t1 = np.mean(v1, axis=1)
#     v0 = v0 - t0.reshape(3, 1)
#     v1 = v1 - t1.reshape(3, 1)
# 
#     if usesvd:
#         # Singular Value Decomposition of covariance matrix
#         u, s, vh = np.linalg.svd(np.dot(v1, v0.T))
#         # rotation matrix from SVD orthonormal bases
#         R = np.dot(u, vh)
#         if np.linalg.det(R) < 0.0:
#             # R does not constitute right handed system
#             R -= np.outer(u[:, 2], vh[2, :]*2.0)
#             s[-1] *= -1.0
#         # homogeneous transformation matrix
#         M = np.identity(4)
#         M[:3, :3] = R
#     else:
#         # compute symmetric matrix N
#         xx, yy, zz = np.sum(v0 * v1, axis=1)
#         xy, yz, zx = np.sum(v0 * np.roll(v1, -1, axis=0), axis=1)
#         xz, yx, zy = np.sum(v0 * np.roll(v1, -2, axis=0), axis=1)
#         N = ((xx+yy+zz, yz-zy,    zx-xz,    xy-yx),
#              (yz-zy,    xx-yy-zz, xy+yx,    zx+xz),
#              (zx-xz,    xy+yx,   -xx+yy-zz, yz+zy),
#              (xy-yx,    zx+xz,    yz+zy,   -xx-yy+zz))
#         # quaternion: eigenvector corresponding to most positive eigenvalue
#         l, V = np.linalg.eig(N)
#         q = V[:, np.argmax(l)]
#         q /= vector_norm(q) # unit quaternion
#         q = np.roll(q, -1) # move w component to end
#         # homogeneous transformation matrix
#         M = quaternion_matrix(q)
# 
#     # scale: ratio of rms deviations from centroid
#     if scaling:
#         v0 *= v0
#         v1 *= v1
#         M[:3, :3] *= math.sqrt(np.sum(v1) / np.sum(v0))
# 
#     # translation
#     M[:3, 3] = t1
#     T = np.identity(4)
#     T[:3, 3] = -t0
#     M = np.dot(M, T)
#     return M


def inverse_matrix(matrix):
    """Return inverse of square transformation matrix.

    >>> M0 = random_rotation_matrix()
    >>> M1 = inverse_matrix(M0.T)
    >>> np.allclose(M1, np.linalg.inv(M0.T))
    True
    >>> for size in range(1, 7):
    ...     M0 = np.random.rand(size, size)
    ...     M1 = inverse_matrix(M0)
    ...     if not np.allclose(M1, np.linalg.inv(M0)): print size

    """
    return np.linalg.inv(matrix)
