import numpy as np
import vedo

from mik_tools import tr, matrix_to_pose, pose_to_matrix, transform_matrix_inverse, eye_pose, transform_points_3d, transform_vectors_3d
from mik_tools.aux.package_utils import get_mesh_path


def get_default_viz(viz=None):
    if viz is None:
        viz = vedo.Plotter()
    return viz


def get_object_mesh(mesh_path:str, scale=1.0):
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


def draw_plane(w_X_pf, color=None, size_x=1, size_y=1, viz=None, alpha=1.0):
    # w_X_pf: (4, 4) as the pose of the plane frame w.r.t the world frame
    viz = get_default_viz(viz)
    obj = vedo.Plane(pos=w_X_pf[:3, 3], normal=w_X_pf[:3, 2], s=(size_x, size_y), alpha=alpha)
    if color is not None:
        obj.color(c=color)
    viz += [obj]
    return viz


def draw_cone(w_X_cf, tan_half_angle, viz=None, color='g', alpha=1.0, scale=1.0):
    # draw a cone with normal axis in the z direction
    # half_angle: angle between the cone edge and the z axis
    # for friction cones, half_angle = np.arctan(mu)
    viz = get_default_viz(viz)
    cone_h = scale
    cone_r = scale*tan_half_angle
    cone_axis = -w_X_cf[:3, 2]
    cone_vertex = w_X_cf[:3, 3] - cone_h*0.5 * cone_axis# np.array([0, 0, cone_h*0.5])
    cone = vedo.Cone(pos=cone_vertex, r=cone_r, height=cone_h, axis=cone_axis, c=color, alpha=alpha)
    viz += [cone]
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


def draw_points(points, color='r', size=4, viz=None):
    # points: (N, 3) numpy array
    viz = get_default_viz(viz)
    points_geom = vedo.Points(points, c=color, r=size)
    viz += [points_geom]
    return viz


def draw_vectors(points, vectors, color='yellow', thickness=5, viz=None):
    """
    Draw vectors originating from points
    :param points:
    :param vectors:
    :param color:
    :param viz:
    :return:
    """
    viz = get_default_viz(viz)
    arrows_w = vedo.Arrows(points, points+vectors, c=color, thickness=thickness)
    viz += [arrows_w]
    return viz


def draw_lines(star_points, end_points, color='black', thickness=5, viz=None):
    """
    Draw lines connecting star_points to end_points
    :param star_points:
    :param end_points:
    :param color:
    :param viz:
    :return:
    """
    viz = get_default_viz(viz)
    lines_w = vedo.Lines(star_points, end_points, c=color, lw=thickness)
    # pts1 = vedo.Points(star_points).legend("Set 1")
    # pts2 = vedo.Points(end_points).legend("Set 2")
    # lines_w = vedo.Lines(pts1, pts2, c=color, lw=thickness)
    viz += [lines_w]
    return viz


def draw_force(force_w, point=None, viz=None, color='r'):
    viz = get_default_viz(viz)
    if point is None:
        point = np.zeros(3)
    force = force_w[:3]
    viz += [vedo.Arrow(start_pt=point, end_pt=point+force, c=color)]
    return viz





def draw_torque(torque_w, point=None, viz=None, color='r'):
    viz = get_default_viz(viz)
    if point is None:
        point = np.zeros(3)
    torque = torque_w[:3]
    torque_norm = np.linalg.norm(torque)
    if torque_norm < 1e-6:
        return viz
    torque_unit_w = torque / torque_norm
    torque_mesh_axis = np.array([1., 0., 0.])
    # get the rotation axis and angle to align the torque with the x axis
    norm_axis = np.cross(torque_mesh_axis, torque_unit_w)
    axis_dot = np.dot(torque_mesh_axis, torque_unit_w)
    angle = np.arccos(axis_dot)
    if 1-np.abs(axis_dot) < 1e-6:
        if angle >= 0:
            rot_axis = np.array([0., 0., 1.])
            angle = 0.
        else:
            rot_axis = np.array([0., 0., 1.])
            angle = np.array([np.pi])
    else:
        rot_axis = norm_axis / np.linalg.norm(norm_axis)
    # w_pose_of = eye_pose  # matrix_to_pose(torque_w) # TODO: Fix this
    w_pose_of = np.concatenate([point, tr.quaternion_about_axis(angle, rot_axis)])
    scale = 1.0 * torque_norm
    mesh_path = get_mesh_path('torque_arrow')
    viz = draw_mesh_from_path(mesh_path=mesh_path, w_pose_of=w_pose_of, color=color, scale=scale, viz=viz)
    return viz


def draw_wrench(wrench_w, point=None, viz=None):
    force_w = wrench_w[:3]
    torque_w = wrench_w[3:]
    viz = draw_force(force_w, point=point, viz=viz, color='pink')
    viz = draw_torque(torque_w, point=point, viz=viz, color='yellow')
    return viz

