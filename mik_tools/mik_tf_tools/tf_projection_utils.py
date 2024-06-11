import sys
import os
import numpy as np
import mik_tools.mik_tf_tools.transformations as tr


def project_pose_to_plane(pose, plane_axis, plane_point, plane_x_axis, orientation_axis=(0,0,1)):
    """
    Observation: All inputselements are assumed to be expressed in the same reference frame!
    pose: numpy array containing [x, y, z, qx, qy, qz, qw]
    plane_axis: [nx, ny, nz] numpy array describing the plane normal axis
    plane_point: [x, y, z] point of the plane representing the new center of the planar pose.
    plane_x_axis: [x, y, z] axis that sets the reference of the planar cooridnates.
    orientation_axis: [x, y, z] represents the axis of the pose that will be used to compute the orientation.
        # theta is the angle between the orientation_axis of the pose and the plane_x axis
    returns [plane_x, plane_y, theta] representing the pose projected in the plane

    """
    plane_axis = np.asarray(plane_axis)
    plane_x_axis = np.asarray(plane_x_axis)
    plane_point = np.asarray(plane_point)
    w_T_plane = get_w_T_plane(plane_axis, plane_point, plane_x_axis)
    plane_quat = tr.quaternion_from_matrix(w_T_plane)
    plane_T_w = tr.inverse_matrix(w_T_plane)
    # transform pose (in world frame) to the plane reference.
    pos_w_extended = np.insert(pose[:3], 3, 1)  # pose in world coorinates
    planar_pos = (plane_T_w @ pos_w_extended)[:-1]
    # compute theta
    theta = get_theta_between_quats(plane_quat, pose[3:], axis_1=(1,0,0), axis_2=orientation_axis, plane_axis=plane_axis)
    # pack all
    planar_pose = np.array([planar_pos[0], planar_pos[1], theta])
    return planar_pose


def unproject_planar_pose(planar_pose, plane_axis, plane_point, plane_x_axis, orientation_axis=(0,0,1)):
    x, y, theta = planar_pose
    w_T_plane = get_w_T_plane(plane_axis, plane_point, plane_x_axis)
    plane_quat = tr.quaternion_from_matrix(w_T_plane)
    # compute the position of the planar pose in the world frame
    pos_planar_ext = np.concatenate([planar_pose[:2], np.array([0, 1])], axis=0)
    pos_w = (w_T_plane @ pos_planar_ext)[:-1]
    delta_quat = tr.quaternion_about_axis(theta, axis=plane_axis)
    quat = tr.quaternion_multiply(delta_quat, plane_quat)
    # TODO: Consider the orientation axis
    pose = np.concatenate([pos_w, quat], axis=0)
    return pose


def get_theta_between_quats(quat_1, quat_2, axis_1=(0,0,1), axis_2=None, plane_axis=None):
    """
    Compute the angle between two quaternions, each
    If plane_axis provided, then the angle is computed on the plane-projected quaternions.
        quat_1: [qx, qy, qz, qw] initial orientation -- this is taken as theta=0
        quat_2: [qx, qy, qz, qw] target orientation
    """

    axis_1 = np.asarray(axis_1)
    if axis_2 is None:
        axis_2 = axis_1
    else:
        axis_2 = np.asarray(axis_2)
    w_R_t = tr.quaternion_matrix(quat_2)[:3, :3]  # target orientation matrix
    w_R_i = tr.quaternion_matrix(quat_1)[:3, :3]  # initial orientation matrix, this is theta = 0
    x_axis_t = w_R_t @ axis_2
    x_axis_i = w_R_i @ axis_1
    if plane_axis is not None:
        x_axis_t_proj = project_vector_to_plane(x_axis_t, plane_axis, normalize=True)
        x_axis_i_proj = project_vector_to_plane(x_axis_i, plane_axis, normalize=True)
    else:
        x_axis_t_proj = norm_vector(x_axis_t)
        x_axis_i_proj = norm_vector(x_axis_i)
    projected_angle = np.arccos(
        np.inner(x_axis_i_proj, x_axis_t_proj) / (np.linalg.norm(x_axis_i_proj) * np.linalg.norm(x_axis_t_proj)))
    # check the angle sign:
    projected_angle_sign = np.sign(np.inner(np.cross(x_axis_i_proj, x_axis_t_proj), plane_axis))
    projected_angle = projected_angle_sign * projected_angle
    theta = projected_angle
    return theta


def get_w_T_plane(plane_axis, plane_point, plane_x_axis):
    """
    Computes the 4x4 Homogenous transformation matrix describing the plane frame with respect to the world frame
        Observation: All inputselements are assumed to be expressed in the same reference frame! -- world frame !!!!!!
        plane_axis: [nx, ny, nz] numpy array describing the plane normal axis
        plane_point: [x, y, z] point of the plane representing the new center of the planar pose.
        plane_x_axis: [x, y, z] axis that sets the reference of the planar cooridnates.
            * This axis does not need to be perpendicular to plane_axis, if it is not, we will project it to the plane,
                and use its projection as plane x_axis
        returns 4x4 Homogenous transformation matrix descirbing the plane frmae with respect to the world frame
    """
    w_T_plane = np.eye(4)
    plane_axis = norm_vector(plane_axis) # unit vector
    plane_x_axis_w = project_vector_to_plane(plane_x_axis, plane_axis, normalize=True)  # coordinates of the x_axis of the plane coordinates expressed as in world coordinates. Note that it belongs to the plane z=0
    plane_y_axis_w = norm_vector(np.cross(plane_axis, plane_x_axis_w))
    w_R_plane = np.stack([plane_x_axis_w, plane_y_axis_w, plane_axis], axis=1)
    plane_center_w = plane_point
    w_T_plane[:3, :3] = w_R_plane
    w_T_plane[:3, 3] = plane_center_w
    return w_T_plane


def project_vector_to_plane(vector, plane_axis, normalize=False):
    vector_perp = np.inner(vector, plane_axis) * plane_axis # projection of the vector along the axis perpendicular axis
    vector_plane = vector - vector_perp
    if normalize:
        vector_plane = norm_vector(vector_plane)
    return vector_plane


def norm_vector(vector):
    vector_normalized = vector / np.linalg.norm(vector)
    return vector_normalized


def angle_between_vectors(vector_1, vector_2, plane_axis=None):
    # TODO: Finish considering the plane angle.
    # compute angle from vector 1 to vector 2, along the axis perpendicular from vector_1 to vector_2
    vector_1_norm = norm_vector(vector_1)
    vector_2_norm = norm_vector(vector_2)

    # check the angle sign:
    if plane_axis is None:
        # plane_axis = norm_vector(np.cross(vector_1, vector_2))
        v2_parallel_proj = np.inner(vector_1_norm, vector_2_norm)
        v2_perp_proj = vector_2_norm - v2_parallel_proj*vector_1_norm
        norm_v2_proj = np.linalg.norm(v2_perp_proj)
        projected_angle = np.arctan2(norm_v2_proj, v2_parallel_proj)
    else:
        vector_1_norm = project_vector_to_plane(vector_1_norm, plane_axis=plane_axis, normalize=True)
        vector_2_norm = project_vector_to_plane(vector_2_norm, plane_axis=plane_axis, normalize=True)
        v2_parallel_proj = np.inner(vector_1_norm, vector_2_norm)
        v2_perp_proj = vector_2_norm - v2_parallel_proj * vector_1_norm
        norm_v2_proj = np.linalg.norm(v2_perp_proj)*np.sign(np.inner(plane_axis, np.cross(vector_1_norm, vector_2_norm)))
        projected_angle = np.arctan2(norm_v2_proj, v2_parallel_proj)
    theta = projected_angle
    return theta




