import numpy as np
import torch
import pandas as pd
from mik_tools.mik_tf_tools import transformations as tr
from mik_tools.math_tools.quaternion_average import average_quaternions, weighted_average_quaternions


eye_pose = np.array([0., 0., 0., 0., 0., 0., 1.])


def pose_to_matrix(pose):
    """
    Convert a pose in position and quaternion format to a 4x4 transformation matrix
    :param pose: (..., 7) numpy array or tensor where the last dimension is composed by [x, y, z, qx, qy, qz, qw]
    :return: pose_matrix (..., 4, 4) numpy array or tensor
    """
    if torch.is_tensor(pose):
        pos = pose[..., :3]  # (..., 3)
        quat = pose[..., 3:]  # (..., 4)
        pose_matrix = tr.quaternion_matrix_batched_tensor(quat)  # (..., 4, 4)
        pose_matrix[..., :3, 3] = pos
    elif type(pose) == np.ndarray:
        if len(pose.shape) == 1:
            pos, quat = np.split(pose, [3])
            pose_matrix = tr.quaternion_matrix(quat)
            pose_matrix[:3, 3] = pos
        else:
            pos = pose[..., :3] # (..., 3)
            quat = pose[..., 3:] # (..., 4)
            pose_matrix = tr.quaternion_matrix_batched_array(quat) # (..., 4, 4)
            pose_matrix[..., :3, 3] = pos
    else:
        raise TypeError('pose must be a torch tensor or a numpy array, but it is {}'.format(type(pose)))
    return pose_matrix


def matrix_to_pose(pose_matrix):
    """
    Convert a 4x4 homogeneous transformation matrix to a pose in position and quaternion format
    :param pose_matrix: (..., 4, 4) numpy array or tensor
    :return: pose (..., 7) numpy array or tensor where the last dimension is composed by [x, y, z, qx, qy, qz, qw]
    """
    if torch.is_tensor(pose_matrix):
        pos = pose_matrix[..., :3, 3]  # (..., 3)
        quat = tr.quaternion_from_matrix_batched_tensor(pose_matrix)
        pose = torch.cat([pos, quat], dim=-1)
    elif type(pose_matrix) == np.ndarray:
        if len(pose_matrix.shape) == 2:
            pos = pose_matrix[:3, 3]
            quat = tr.quaternion_from_matrix(pose_matrix)
            pose = np.concatenate([pos, quat])
        else:
            pos = pose_matrix[..., :3, 3]
            quat = tr.quaternion_from_matrix_batched_array(pose_matrix)
            pose = np.concatenate([pos, quat], axis=-1)
    else:
        raise TypeError('pose_matrix must be a torch tensor or a numpy array, but it is {}'.format(type(pose_matrix)))
    return pose


def transform_matrix_inverse(matrix):
    """
    Compute the inverse of a 4x4 homogeneous transformation matrix
    :param matrix: numpy array or tensor of shape (..., 4, 4) or (..., 3, 3)
    :return: inverse_matrix: numpy array or tensor of shape (..., 4, 4) or (..., 3, 3)
    """
    # check if it is a 2d transform or a 3d transform
    assert len(matrix.shape) >= 2, 'matrix must have at least 2 dimensions'
    assert matrix.shape[-1] == matrix.shape[-2], 'matrix must be square'
    space_size = matrix.shape[-1] - 1
    # NOTE: the transform matrix inverse given by X = [R | t] is X^-1 = [R^T | -R^T*t]
    R = matrix[..., :space_size, :space_size] # (..., space_size, space_size)
    t = matrix[..., :space_size, space_size] # (..., space_size)
    if isinstance(matrix, np.ndarray):
        R_inv = np.swapaxes(R, -2, -1) # tranpose the last two dimensions
        t_inv = - np.einsum('...ij,...j->...i', R_inv, t)
        hom_part = np.zeros(matrix.shape[:-2]+(1, space_size + 1))
        hom_part[..., 0, -1] = 1
        matrix_inv = np.concatenate([np.concatenate([R_inv, t_inv[..., np.newaxis]], axis=-1), hom_part], axis=-2)
    elif torch.is_tensor(matrix):
        R_inv = R.transpose(-2, -1)
        t_inv = - torch.einsum('...ij,...j->...i', R_inv, t)
        hom_part = torch.zeros(matrix.shape[:-2]+(1, space_size + 1), device=matrix.device, dtype=matrix.dtype)
        hom_part[..., 0, -1] = 1
        matrix_inv = torch.cat([torch.cat([R_inv, t_inv.unsqueeze(-1)], dim=-1), hom_part], dim=-2)
    else:
        raise TypeError('matrix must be a torch tensor or a numpy array, but it is {}'.format(type(matrix)))
    return matrix_inv


def transform_points_3d(points_f1, f2_X_f1, points_have_point_dimension=True, only_points_are_batched=False, only_transform_are_batched=False):
    """
    Transform a set of points in 2D from frame f1 to frame f2
    :param points_f1: tensor or array of shape (..., K, 3) if have_point_dimension is True, or (..., 3) if have_point_dimension is False
    :param f2_X_f1: tensor or array or shape (..., 4, 4) or (4,4)
    :param points_have_point_dimension: bool indicating if points have an extra dimension for the points
    :return: points_f2 tensor or array of shape (..., K, 3) if have_point_dimension is True, or (..., 3) if have_point_dimension is False
    """
    if points_have_point_dimension:
        if only_points_are_batched:
            einusm_key = 'ij,...kj->...ki'
        elif only_transform_are_batched:
            einusm_key = '...ij,kj->...ki'
        else:
            # both are batched
            einusm_key = '...ij,...kj->...ki'
    else:
        if only_points_are_batched:
            einusm_key = 'ij,...j->...i'
        elif only_transform_are_batched:
            einusm_key = '...ij,j->...i'
        else:
            # both are batched
            einusm_key = '...ij,...j->...i'
    if isinstance(points_f1, torch.Tensor):
        points_f1_hom = torch.cat(
            [points_f1, torch.ones_like(points_f1[..., :1], dtype=points_f1.dtype, device=points_f1.device)], dim=-1)
        points_f2_hom = torch.einsum(einusm_key, f2_X_f1, points_f1_hom)  # (..., K, 4)
    elif isinstance(points_f1, np.ndarray):
        points_f1_hom = np.concatenate([points_f1, np.ones_like(points_f1[..., :1])], axis=-1)
        points_f2_hom = np.einsum(einusm_key, f2_X_f1, points_f1_hom)  # (..., K, 4)
    else:
        raise ValueError('Unsupported type')
    points_f2 = points_f2_hom[..., :3] / points_f2_hom[..., 3:]
    return points_f2


def transform_vectors_3d(vectors_f1, f2_X_f1, vectors_have_point_dimension=False):
    """
    Transform a set of vectors in 2D from frame f1 to frame f2
    :param vectors_f1: tensor or array of shape (..., 3) if vectors_have_point_dimension is False, or (..., K, 3) if vectors_have_point_dimension is True
    :param f2_X_f1: tensor or array or shape (..., 4, 4) or (4,4)
    :param vectors_have_point_dimension: bool indicating if vectors have an extra dimension for the points
    :return: vectors_f2 tensor or array of shape (..., 3) if vectors_have_point_dimension is False, or (...,K, 3) if vectors_have_point_dimension is True
    """
    if vectors_have_point_dimension:
        einsum_key = '...ij,...kj->...ki'
    else:
        einsum_key = '...ij,...j->...i'
    if isinstance(vectors_f1, torch.Tensor):
        vectors_f2 = torch.einsum(einsum_key, f2_X_f1[..., :3, :3], vectors_f1)  # (..., K, 3)
    elif isinstance(vectors_f1, np.ndarray):
        vectors_f2 = np.einsum(einsum_key, f2_X_f1[..., :3, :3], vectors_f1) # (..., K, 3)
    else:
        raise ValueError('Unsupported type')
    return vectors_f2


# ======================================================================================================================
# ================================== 2D TRANSFORMATIONS ================================================================
# ======================================================================================================================

def pose_to_matrix_2d(pose_2d):
    """
    pose is composed by [x, y, theta]
    matrix is composed by [[R, t], [0, 1]] where R is a 2x2 rotation matrix and t is a 2x1 translation vector
    :param pose_2d: np.array or torch tensor of shape (..., 3) composed by [x, y, theta]
    :return: pose_matrix_2d: np.array or torch tensor of shape (..., 3, 3)
    """
    if torch.is_tensor(pose_2d):
        pose_matrix_2d = pose_to_matrix_2d_tensor(pose_2d)
    elif isinstance(pose_2d, np.ndarray):
        pose_matrix_2d = pose_to_matrix_2d_array(pose_2d)
    else:
        raise TypeError('pose_2d must be a torch tensor or a numpy array, but it is {}'.format(type(pose_2d)))
    return pose_matrix_2d


def matrix_to_pose_2d(pose_matrix_2d):
    """
    pose is composed by [x, y, theta]
    matrix is composed by [[R, t], [0, 1]] where R is a 2x2 rotation matrix and t is a 2x1 translation vector
    :param pose_matrix_2d: np.array or torch tensor of shape (..., 3, 3)
    :return: 2d_pose: np.array or torch tensor of shape (..., 3) composed by [x, y, theta]
    """
    if torch.is_tensor(pose_matrix_2d):
        pose_2d = matrix_to_pose_2d_tensor(pose_matrix_2d)
    elif type(pose_matrix_2d) == np.ndarray:
        pose_2d = matrix_to_pose_2d_array(pose_matrix_2d)
    else:
        raise TypeError('pose_2d must be a torch tensor or a numpy array, but it is {}'.format(type(pose_matrix_2d)))
    return pose_2d


def get_2d_rotation_matrix(angle):
    """

    :param angle: (...) numpy or tensor
    :return: (...,2,2) rotation matrix
    """
    if torch.is_tensor(angle):
        R = get_2d_rotation_matrix_tensor(angle)
    elif isinstance(angle, np.ndarray):
        R = get_2d_rotation_matrix_array(angle)
    else:
        raise TypeError('angle must be a torch tensor or a numpy array, but it is {}'.format(type(angle)))
    return R



# ======================================================================================================================
# ======================================================================================================================



# ======================================================================================================================
# ================================== TF FRAME  ================================================================
# ======================================================================================================================

def get_transformation_matrix(all_tfs, source_frame, target_frame):
    w_X_sf = all_tfs[source_frame]
    w_X_tf = all_tfs[target_frame]
    sf_X_w = transform_matrix_inverse(w_X_sf)
    sf_X_tf = sf_X_w @ w_X_tf
    return sf_X_tf


def average_poses(poses, weights=None):
    # poses: Nx7 numpy matrix containing poses to average in the rows.
    # The poses are arranged as (x,y,z,qx,qy,qz,qw), with (x,y,z) being the position and (qx,qy,qz,qw) the quaternion
    # weights: Nx1 numpy matrix containing the weights
    # The result will be the average pose of the input. Note that the signs of the output quaternion can be reversed, since q and -q describe the same orientation
    pos = poses[:, :3]
    quats = poses[:, 3:]
    if weights is None:
        weights = np.ones((poses.shape[0], 1))/poses.shape[0]
    quat_avg = weighted_average_quaternions(quats, weights)
    pos_avg = np.average(pos, axis=0, weights=weights.flatten())
    pose_avg = np.concatenate([pos_avg, quat_avg])
    return pose_avg

def convert_all_tfs_to_tensors(all_tfs):
    """
       Convert a DataFrame object containing the tfs with respect a common frame into a dictionary of tensor transformation matrices
       :param all_tfs: DataFrame
       :return:
       """
    # Transform a DF into a dictionary of homogeneous transformations matrices (4x4)
    converted_all_tfs = {}
    parent_frame = all_tfs['parent_frame'].iloc[0]  # Assume that are all the same
    child_frames = all_tfs['child_frame']
    converted_all_tfs[parent_frame] = np.eye(4)  # Transformation to itself is the identity
    all_poses = all_tfs[['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']]
    for i, child_frame_i in enumerate(child_frames):
        pose_i = all_poses.iloc[i]
        X_i = tr.quaternion_matrix(pose_i[3:])
        X_i[:3, 3] = pose_i[:3]
        converted_all_tfs[child_frame_i] = X_i
    return converted_all_tfs


def pack_pose_to_tf_dict(pose, parent_frame, child_frame):
    tf_dict = {
        'frame_id': parent_frame,
        'child_id': child_frame,
        'x': pose[0],
        'y': pose[1],
        'z': pose[2],
        'qx': pose[3],
        'qy': pose[4],
        'qz': pose[5],
        'qw': pose[6],
    }
    return tf_dict


def extract_tfs_from_dataframe(tfs_df, frame_id=None, ref_id=None, tf_column_names=None):
    # add fake tr from parent_frame to parent frame
    parent_frames = tfs_df['parent_frame'].values
    if tf_column_names is None:
        tf_column_names = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
    unique_parent_frames = np.unique(parent_frames)
    id_tr = [0., 0., 0., 0., 0., 0., 1.] # identity transformation -- ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
    parent_frames_data = []
    for parent_frame in unique_parent_frames:
        data_line = [parent_frame, parent_frame] + id_tr
        parent_frames_data.append(data_line)
    parent_frames_df = pd.DataFrame(parent_frames_data, columns=['parent_frame', 'child_frame']+tf_column_names)
    tfs_df = pd.concat([tfs_df, parent_frames_df], ignore_index=True)
    # check the child frames for the frame_id
    frame_ids = tfs_df['child_frame'].values # all frame ids
    if frame_id is None:
        # load tf all data
        tfs = tfs_df[tf_column_names].values
    elif frame_id in frame_ids:
        # return only the tf for the given child frame
        tfs = tfs_df[tfs_df['child_frame'] == frame_id][tf_column_names].values
    else:
        # frame not found
        raise NameError('No frame named {} not found. Available frames: {}'.format(frame_id, frame_ids))
    # here tfs are
    if ref_id is not None:
        tr_tfs = []
        # transform the tfs so they are expressed with respect to the frame ref_id
        if ref_id not in frame_ids:
            raise NameError(f'Reference frame named {ref_id} not found found. Available frames: {frame_ids}')
        w_X_ref_id = pose_to_matrix(tfs_df[tfs_df['child_frame'] == ref_id][tf_column_names].values[0]) # TODO: Extenc this for different world reference frames
        for tf_i in tfs:
            w_X_frame_id = pose_to_matrix(tf_i)
            ref_id_X_frame_id = transform_matrix_inverse(w_X_ref_id) @ w_X_frame_id
            tr_tfs.append(matrix_to_pose(ref_id_X_frame_id))
        tfs = np.stack(tr_tfs, axis=0)
    return tfs







# ======================================================================================================================
# ================================== AUXILIARY FUNCTIONS ===============================================================
# ======================================================================================================================


def pose_to_matrix_2d_tensor(pose_2d):
    """
    :param pose_2d: torch tensor (..., 3) where the last dimension is composed by [x, y, theta]
    :return: matrix_2d: torch tensor  (..., 3, 3) where the last two dimensions are the 2D transformation matrix
    """
    batch_dims = pose_2d.shape[:-1]
    pose_matrix_2d = torch.eye(3, device=pose_2d.device, dtype=pose_2d.dtype)
    if len(batch_dims) > 0:
        pose_matrix_2d = pose_matrix_2d.unsqueeze(0).repeat(batch_dims+(1,1))
    pos_xys = pose_2d[..., :2]
    theatas = pose_2d[..., 2]
    R = get_2d_rotation_matrix_tensor(theatas)
    pose_matrix_2d[..., :2, 2] = pos_xys
    pose_matrix_2d[..., :2, :2] = R
    return pose_matrix_2d


def matrix_to_pose_2d_tensor(pose_matrix_2d):
    """
    pose is composed by [x, y, theta]
    matrix is composed by [[R, t], [0, 1]] where R is a 2x2 rotation matrix and t is a 2x1 translation vector
    :param pose_matrix_2d: torch tensor (..., 3, 3)
    :return: pose_2d: torch tensor (..., 3) composed by [x, y, theta]
    """
    batch_dims = pose_matrix_2d.shape[:-2]
    pos_2d = pose_matrix_2d[..., :2, 2]
    theta = torch.atan2(pose_matrix_2d[..., 1, 0], pose_matrix_2d[..., 0, 0])
    pose_2d = torch.cat([pos_2d, theta.unsqueeze(-1)], dim=-1)
    return pose_2d


def matrix_to_pose_2d_array(pose_matrix_2d):
    """
    pose is composed by [x, y, theta]
    :param pose_matrix_2d: numpy array of shape (..., 3, 3)
    :return:
    """
    batch_dims = pose_matrix_2d.shape[:-2]
    pos_2d = pose_matrix_2d[..., :2, 2] # (..., 2)
    theta = np.arctan2(pose_matrix_2d[..., 1, 0], pose_matrix_2d[..., 0, 0]) # (..., 2)
    pose_2d = np.concatenate([pos_2d, theta[..., np.newaxis]], axis=-1)
    return pose_2d


def pose_to_matrix_2d_array(pose_2d):
    """
    :param pose_2d: np.array (..., 3) where the last dimension is composed by [x, y, theta]
    :return: matrix_2d: np.array (..., 3, 3) where the last two dimensions are the 2D transformation matrix
    """
    batch_dims = pose_2d.shape[:-1]
    pose_matrix_2d = np.eye(3)
    if len(batch_dims) > 0:
        # repeat the matrix for each batch
        pose_matrix_2d = pose_matrix_2d[np.newaxis, ...].repeat(np.prod(batch_dims), axis=0).reshape(batch_dims+(3,3)) # (..., 3, 3)
    pos_xys = pose_2d[..., :2]
    theatas = pose_2d[..., 2]
    R = get_2d_rotation_matrix_array(theatas) # (..., 2, 2)
    pose_matrix_2d[..., :2, 2] = pos_xys
    pose_matrix_2d[..., :2, :2] = R
    return pose_matrix_2d


def get_2d_rotation_matrix_tensor(angle):
    """

    :param angle: (...) tensor
    :return: (...,2,2) rotation matrix
    """
    R = torch.eye(2, device=angle.device, dtype=angle.dtype)
    if len(angle.shape) > 0:
        R = R.unsqueeze(0).repeat(angle.shape+(1,1))
    cos_thetas = torch.cos(angle)
    sin_thetas = torch.sin(angle)
    R[..., 0, 0] = cos_thetas
    R[..., 1, 1] = cos_thetas
    R[..., 0, 1] = -sin_thetas
    R[..., 1, 0] = sin_thetas
    return R


def get_2d_rotation_matrix_array(angle):
    """

    :param angle: (...) numpy
    :return: (...,2,2) rotation matrix
    """
    R = np.eye(2)
    if len(angle.shape) > 0:
        R = R[np.newaxis, ...].repeat(np.prod(angle.shape), axis=0).reshape(angle.shape+(2,2))
    cos_thetas = np.cos(angle)
    sin_thetas = np.sin(angle)
    R[..., 0, 0] = cos_thetas
    R[..., 1, 1] = cos_thetas
    R[..., 0, 1] = -sin_thetas
    R[..., 1, 0] = sin_thetas
    return R


