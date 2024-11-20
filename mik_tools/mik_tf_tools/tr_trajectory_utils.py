import numpy as np
from mik_tools.mik_tf_tools import transformations as tr
from mik_tools.mik_tf_tools.tf_tools import matrix_to_pose, pose_to_matrix


def project_pose_to_plane(pose, w_T_plane, ff_quat=None, gw_axis=None):
    """
    Projects SE(3) pose to SE(2) pose on a plane.
    :param pose: SE(3) pose in world coordinates in format [x, y, z, qx, qy, qz, qw]
    :param w_T_plane: homogeneous transformation matrix from plane coordinates to world coordinates
    :param ff_quat: orientation of the frame at the plaen origin, i.e. at plane coorinates [0, 0, 0]
    :return: SE(2) pose in plane coordinates in format [x, y, theta]
    """
    if ff_quat is None:
        ff_quat = np.array([0, 0, 0, 1]) # [qx, qy, qz, qw]
    plane = w_T_plane[:3, :3] @ np.array([0, 0, 1.]) # plane normal axis
    theta = get_theta_between_quats(ff_quat, pose[3:], plane, x_axis=gw_axis)
    plane_T_w = tr.inverse_matrix(w_T_plane)
    pos_w_extended = np.insert(pose[:3], 3, 1)  # pose in world coorinates
    planar_pos = (plane_T_w @ pos_w_extended)[:-1]
    planar_pose = np.array([planar_pos[0], planar_pos[1], theta])
    return planar_pose


def unproject_planar_pose(planar_pose, w_T_plane, ff_quat=None):
    """
    Projects a SE(2) planar pose to a pose to SE(3)
    :param planar_pose: SE(2) pose in world coordinates in format [x, y, theta]
    :param w_T_plane: homogeneous transformation matrix from plane coordinates to world coordinates
    :param ff_quat: orientation of the frame at the plaen origin, i.e. at plane coorinates [0, 0, 0]
    :return: SE(3) pose in plane coordinates in format [x, y, z, qx, qy, qz, qw]
    """
    if ff_quat is None:
        ff_quat = np.array([0, 0, 0, 1]) # [qx, qy, qz, qw]
    x, y, theta = planar_pose
    pos_planar_ext = np.concatenate([planar_pose[:2], np.array([0, 1])], axis=0)
    pos_w = (w_T_plane @ pos_planar_ext)[:-1]
    plane = w_T_plane[:3, :3] @ np.array([0, 0, 1.]) # plane normal axis
    delta_quat = tr.quaternion_about_axis(theta, axis=plane)
    quat = tr.quaternion_multiply(delta_quat, ff_quat)
    pose = np.concatenate([pos_w, quat], axis=0)
    return pose


def get_w_T_plane(point, plane, quat=None, x_axis=None):
    # point: plane origin point in the world coordinates
    # plane: plane normal vector in the world coordinates
    # w_T_plane: homogeneous transformation matrix from plane coordinates to world coordinates
    # i.e.  pose_w = w_T_plane @ pose_plane
    z_axis = np.array([0, 0, 1])
    w_T_plane = np.eye(4)
    if quat is None:
        plane = plane / np.linalg.norm(plane)
        # Compute the x axis of the plane coordinates expressed as in world coordinates
        if x_axis is None:
            if np.dot(plane, z_axis) == 1: # they are colinear
                plane_x_axis_w = np.array([1, 0, 0])
            else:
                _plane_x_axis_w = np.cross(z_axis,
                                           plane)  # coordinates of the x_axis of the plane coordinates expressed as in world coordinates. Note that it belongs to the plane z=0
                plane_x_axis_w = _plane_x_axis_w / np.linalg.norm(_plane_x_axis_w)
        else:
            plane_x_axis_w = x_axis / np.linalg.norm(x_axis)
        # Compute the y axis of the plane coordinates expressed as in world coordinates
        _plane_y_axis_w = np.cross(plane, plane_x_axis_w)
        plane_y_axis_w = _plane_y_axis_w / np.linalg.norm(_plane_y_axis_w)
        w_R_plane = np.stack([plane_x_axis_w, plane_y_axis_w, plane], axis=1)
    else:
        w_R_plane = tr.quaternion_matrix(quat)[:3, :3]
    plane_center_w = point
    w_T_plane[:3, :3] = w_R_plane
    w_T_plane[:3, 3] = plane_center_w
    return w_T_plane


def get_theta_between_quats(quat_1, quat_2, plane, x_axis=None):
    """
    Get the angle between two quaternions projected on a plane.
    :param quat_1: quaternion 1 in format [qx, qy, qz, qw]
    :param quat_2: quaternion 2 in format [qx, qy, qz, qw]
    :param plane: plane normal vector in the world coordinates
    :param gf_axis: axis of the gripper frame in the world coordinates
    """
    if x_axis is None:
        x_axis = np.array([1, 0, 0])
    w_R_t = tr.quaternion_matrix(quat_2)[:3, :3]  # target orientation matrix
    w_R_i = tr.quaternion_matrix(quat_1)[:3, :3]  # initial orientation matrix, this is theta = 0
    x_axis_t = w_R_t @ x_axis
    x_axis_i = w_R_i @ x_axis
    _x_axis_t_proj = x_axis_t - np.inner(x_axis_t, plane) * plane
    _x_axis_i_proj = x_axis_i - np.inner(x_axis_i, plane) * plane
    x_axis_t_proj = _x_axis_t_proj / np.linalg.norm(_x_axis_t_proj)
    x_axis_i_proj = _x_axis_i_proj / np.linalg.norm(_x_axis_i_proj)
    projected_angle = np.arccos(
        np.inner(x_axis_i_proj, x_axis_t_proj) / (np.linalg.norm(x_axis_i_proj) * np.linalg.norm(x_axis_t_proj)))
    # check the angle sign:
    projected_angle_sign = np.sign(np.inner(np.cross(x_axis_i_proj, x_axis_t_proj), plane))
    projected_angle = projected_angle_sign * projected_angle
    theta = projected_angle
    return theta


def transform_pose(pose, trans=None, quat=None):
    # pose: [x, y, z, qx, qy, qz, qw]
    if trans is None:
        trans = np.zeros(3)
    if quat is None:
        quat = np.array([0, 0, 0, 1])
    pose_tr = np.concatenate([trans, quat])
    T_tr = pose_to_matrix(pose_tr)
    T_current = pose_to_matrix(pose)
    T_out = T_current @ T_tr
    pose_out = matrix_to_pose(T_out)
    return pose_out


def rotate_planar_pose(planar_pose, angle, point=None):
    if point is None:
        point = np.array([0., 0.])
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    rot_planar_pos = point + R @ (planar_pose[:2] - point)
    rot_angle = planar_pose[-1] + angle
    rot_planar_pose = np.insert(rot_planar_pos, len(rot_planar_pos), rot_angle)
    return rot_planar_pose


def rotation_along_point_angle_2d(pose_init, angle, point=None, max_angle_step=None, num_steps=1):
    # pose_init: [x, y, theta]
    if point is None:
        point = np.array([0., 0.])
    if max_angle_step is not None:
        num_steps = int(abs(angle//max_angle_step))
    angle_step = angle/num_steps # distribute the points evently
    angles = np.sign(angle)*np.linspace(np.abs(angle_step), np.abs(angle), num_steps)
    poses = [rotate_planar_pose(pose_init, angle, point) for angle in angles] # 2d poses
    return poses


def pose_matrix_trajectory_interpolation(w_X_init:np.ndarray, w_X_final:np.ndarray, num_steps:int) -> np.ndarray:
    """
    Interpolates between two SE(3) poses.
    :param w_X_init: (4, 4) as the initial pose matrix
    :param w_X_final: (4, 4) as the final pose matrix
    :param num_steps: (int) as the number of steps to interpolate
    :return: w_X_interpolated: (num_steps+1, 4, 4) as the interpolated poses
    """
    w_X_interpolated = []
    for i in range(num_steps):
        alpha = i / num_steps
        w_X_i = pose_matrix_interpolation(w_X_init, w_X_final, alpha)
        w_X_interpolated.append(w_X_i)
    w_X_interpolated.append(pose_matrix_interpolation(w_X_init, w_X_final, 1.)) # Add the last pose
    w_X_interpolated = np.stack(w_X_interpolated, axis=0) # (num_steps+1, 4, 4)
    return w_X_interpolated


def pose_matrix_interpolation(w_X_init, w_X_final, alpha):
    """
    Interpolates between two SE(3) poses.
    :param w_X_init: (4, 4) as the initial pose matrix
    :param w_X_final: (4, 4) as the final pose matrix
    :param alpha: (float) as the interpolation factor (between 0 and 1)
    :return: w_X_interpolated: (num_steps, 4, 4) as the interpolated poses
    """
    w_pose_init = matrix_to_pose(w_X_init)
    w_pose_final = matrix_to_pose(w_X_final)
    quat_init = w_pose_init[3:]
    quat_final = w_pose_final[3:]
    quat_i = tr.quaternion_slerp(quat_init, quat_final, alpha)
    pos_i = w_pose_init[:3] + alpha * (w_pose_final[:3] - w_pose_init[:3])
    w_X_i = pose_to_matrix(np.concatenate([pos_i, quat_i]))
    return w_X_i
