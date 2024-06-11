import numpy as np
import open3d as o3d

from mik_tools.mik_tf_tools.tr_trajectory_utils import project_pose_to_plane, get_w_T_plane, unproject_planar_pose, \
    get_theta_between_quats, transform_pose, rotation_along_point_angle_2d
import  mik_tools.mik_tf_tools.transformations as tr
from mik_tools.math_tools.geometry_tools import get_angle_between_vectors


def get_frames(poses, scale=1.0):
    mesh_frames = []
    for pose in poses:
        pos = pose[:3]
        quat = pose[3:]
        mesh_frame_i = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale,
                                                                       origin=[0, 0, 0])
        R = tr.quaternion_matrix(quat)[:3,:3]
        mesh_frame_i.rotate(R, center=(0, 0, 0))
        mesh_frame_i.translate(pos)
        mesh_frames.append(mesh_frame_i)
    return mesh_frames


def get_plane(plane, point, plane_size=1.):
    plane_w = 2.*plane_size
    plane_h = 2.*plane_size
    plane_thick = 0.001
    plane_mesh = o3d.geometry.TriangleMesh.create_box(width=plane_w, height=plane_h, depth=plane_thick) # plane (0,0,1)
    plane_center = np.array([-0.5*plane_w, -0.5*plane_h, -plane_thick])
    plane_mesh.translate(plane_center) # plane offset
    current_plane_axis = np.array([0, 0, 1])
    angle, norm_vector = get_angle_between_vectors(current_plane_axis, plane)
    quat = tr.quaternion_about_axis(angle, norm_vector)
    R = tr.quaternion_matrix(quat)[:3,:3]
    plane_mesh.rotate(R, center=(0, 0, 0))
    plane_mesh.translate(point)
    plane_mesh.paint_uniform_color([.9, 0.9, 0.9])
    return plane_mesh


def view_planar_poses(target_planar_poses, point, plane, axis=None, ff_quat=None):
    w_T_plane = get_w_T_plane(point=point, plane=plane)
    poses = [unproject_planar_pose(planar_pose, w_T_plane, ff_quat) for planar_pose in target_planar_poses]
    poses += [unproject_planar_pose(np.zeros(3), w_T_plane, ff_quat)] # include the plane origin
    mesh_frames = get_frames(poses, scale=0.1)
    plane_mesh = get_plane(plane, point)
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.2)
    o3d.visualization.draw_geometries(mesh_frames + [origin_frame, plane_mesh])



# DEBUG:



def test_1():
    # simple rotation ------------------------------------
    # plane
    plane = np.array([1, 0, 0])  # plane normal axis
    point = np.array([1, 0, 0])
    ff_quat = np.array([0, 0, 0, 1])  # qx, qy, qz, qw

    # (x, y z)
    # planar_poses = np.array([
    #     [0,0,0],
    #     [1,0,-np.pi*0.25],
    #     # [1,0,0.],
    #     [0,1, np.pi*0.25],
    #     # [0,1,0.],
    #     [-1,-1, 0],
    # ])
    pose_init = np.array([1, 0, 0])
    planar_poses = [np.zeros(3), pose_init] + rotation_along_point_angle_2d(pose_init, angle=np.pi,
                                                                            point=np.array([0, 0]), num_steps=20)

    view_planar_poses(planar_poses, point, plane, ff_quat=ff_quat)


def test_2():
    # double plane ------------------------------------
    # plane
    plane = np.array([1, 0, 0])  # plane normal axis
    point = np.array([1, 0, 0])
    ff_quat_des = np.array([0, 0, np.sin(0.5 * np.pi * 0.5), np.cos(0.5 * np.pi * 0.5)])  # qx, qy, qz, qw
    ff_quat = np.array([0, 0, 0, 1])  # qx, qy, qz, qw

    # (x, y z)
    # planar_poses = np.array([
    #     [0,0,0],
    #     [1,0,-np.pi*0.25],
    #     # [1,0,0.],
    #     [0,1, np.pi*0.25],
    #     # [0,1,0.],
    #     [-1,-1, 0],
    # ])
    pose_init = np.array([1, 0, 0])
    planar_poses = [np.zeros(3), pose_init] + rotation_along_point_angle_2d(pose_init, angle=np.pi,
                                                                            point=np.array([0, 0]), num_steps=20)

    w_T_plane = get_w_T_plane(point=point, plane=plane)
    poses = [unproject_planar_pose(planar_pose, w_T_plane, ff_quat_des) for planar_pose in planar_poses]

    # poses = [transform_pose(pose, quat=ff_quat_des) for pose in poses]

    # transform the poses back to the plane:
    plane_poses_proj = [project_pose_to_plane(pose, w_T_plane, ff_quat_des) for pose in poses]
    # put the poses on another plane:
    point_2 = np.array([0, 0, 0])
    plane_2 = np.array([0, 1, 0])
    w_T_plane_2 = get_w_T_plane(point=point_2, plane=plane_2)
    poses_2 = [unproject_planar_pose(planar_pose, w_T_plane_2, ff_quat_des) for planar_pose in plane_poses_proj]

    # poses += [np.array([0, 1, 1, 0, 0, np.sin(np.pi*0.5*0.5), np.cos(np.pi*0.5*0.5)])]
    mesh_frames = get_frames(poses, scale=0.1)
    plane_mesh = get_plane(plane, point)

    mesh_frames_2 = get_frames(poses_2, scale=0.1)
    plane_mesh_2 = get_plane(plane_2, point_2)

    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.2)

    o3d.visualization.draw_geometries(mesh_frames + [origin_frame, plane_mesh, plane_mesh_2] + mesh_frames_2)


def test_3():
    # negative rotation ------------------------------------
    plane = np.array([1, 0, 0])  # plane normal axis
    point = np.array([1, 0, 0])
    ff_quat = np.array([0, 0, 0, 1])  # qx, qy, qz, qw
    pose_init = np.array([1, 0, 0])
    planar_poses = [np.zeros(3), pose_init] + rotation_along_point_angle_2d(pose_init, angle=-np.pi,
                                                                            point=np.array([0, 0]), num_steps=20)

    view_planar_poses(planar_poses, point, plane, ff_quat=ff_quat)


if __name__ == '__main__':
    test_1()
    test_2()
    test_3()






