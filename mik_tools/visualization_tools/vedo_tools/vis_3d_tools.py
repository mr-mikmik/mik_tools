import numpy as np
import vedo

from mik_tools import tr, matrix_to_pose, pose_to_matrix, transform_matrix_inverse, eye_pose, transform_points_3d, transform_vectors_3d


def get_default_viz(viz=None):
    if viz is None:
        viz = vedo.Plotter()
    return viz


def get_object_mesh(mesh_path, scale=1.0):
    obj_mesh = vedo.Mesh(mesh_path)
    obj_mesh = obj_mesh.scale(s=scale)
    return obj_mesh


def draw_mesh(obj_mesh_of, w_pose_of, color=None, alpha=1.0, viz=None):
    viz = get_default_viz(viz)
    # paint the mesh
    if color is not None:
        obj_mesh_of.color(c=color, alpha=alpha)
    # transform it to the pose
    w_X_of = pose_to_matrix(w_pose_of)
    obj_mesh_w = obj_mesh_of.apply_transform(w_X_of)
    viz += [obj_mesh_w]
    return viz


def draw_mesh_from_path(mesh_path, w_pose_of, color=None, alpha=1.0, scale=1.0, viz=None):
    obj_mesh_of = get_object_mesh(mesh_path, scale=scale)
    viz = draw_mesh(obj_mesh_of, w_pose_of, color=color, alpha=alpha, viz=viz)
    return viz


def draw_box(box_size, w_pose_of, color=None, alpha=1.0, viz=None):
    viz = get_default_viz(viz)
    obj = vedo.Box(size=box_size)
    if color is not None:
        obj.color(c=color, alpha=alpha)
    # transform it to the pose
    w_X_of = pose_to_matrix(w_pose_of)
    obj_mesh_w = obj.apply_transform(w_X_of)
    viz += [obj_mesh_w]
    return viz


def draw_frame(pose, scale=1.0, viz=None):
    viz = get_default_viz(viz)
    arrow_ends = scale*np.eye(3) # in frame reference
    w_pose_arrows = None
    colors = ['r', 'g', 'b']
    # get the arrows in world frame
    w_X_f = pose_to_matrix(pose)
    arrow_ends_w = transform_points_3d(arrow_ends, w_X_f) # (3, 3)
    start_p = pose[:3]
    arrows_geoms = []
    for i, color_i in enumerate(colors):
        arrow_i = vedo.Arrow(start_pt=start_p, end_pt=arrow_ends_w[i], c=color_i)
        arrows_geoms.append(arrow_i)
    viz += arrows_geoms
    return viz


def draw_points(points, color='r', viz=None):
    # points: (N, 3) numpy array
    viz = get_default_viz(viz)
    points_geom = vedo.Points(points, c=color)
    viz += points_geom
    return viz

