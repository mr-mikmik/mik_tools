import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import Image as IPythonImage

from mik_tools.data_utils.data_path_tools import get_output_trajectory_gif_path


def view_points(points_xyz, colors=None, marker_size=None, visible_axis=True, fig=None):
    # points_xyz of shape (N, 3)
    if colors is None:
        colors = np.zeros_like(points_xyz)
    elif len(colors.shape) == 1:
        # same color for all
        colors = np.tile(colors, (len(points_xyz), 1))
    if marker_size is None:
        marker_size = np.ones((len(points_xyz)))
    elif type(marker_size) in [int, float]:
        marker_size = np.ones((len(points_xyz))) * marker_size
    marker_size = marker_size.astype(np.int64)
    # make sure that all colors are in the range [0, 255]
    if np.all(colors <= 1):
        colors = colors * 255
    color_list = [f'rgb({color_i[0]}, {color_i[1]}, {color_i[2]})' for color_i in colors]
    if fig is None:
        fig = px.scatter_3d(x=points_xyz[:, 0], y=points_xyz[:, 1], z=points_xyz[:, 2])
    else:
        fig.add_trace(go.Scatter3d(x=points_xyz[:, 0], y=points_xyz[:, 1], z=points_xyz[:, 2]))
    fig.update_scenes(aspectmode='data')
    #     fig.update_traces(marker_size=marker_size)
    fig.update_traces(mode='markers', marker=dict(color=color_list, size=list(marker_size), line=dict(width=0)))
    if not visible_axis:
        fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    return fig


def view_points_groups(points_xyz_list, colors_list, marker_size_list, visible_axis=True):
    num_points = [len(points_i) for points_i in points_xyz_list]
    colors = []
    marker_sizes = []
    for i, color_i in enumerate(colors_list):
        num_points_i = num_points[i]
        marker_size_i = marker_size_list[i]
        if color_i is None:
            color_i = np.zeros((num_points_i, 3))
        elif len(color_i.shape) == 1:
            # same color for all
            color_i = np.tile(color_i, (num_points_i, 1))
        colors.append(color_i)
        if marker_size_i is None:
            marker_size_i = np.ones((num_points_i))
        elif type(marker_size_i) in [int, float]:
            marker_size_i = np.ones((num_points_i)) * marker_size_i
        marker_sizes.append(marker_size_i)
    points_xyz = np.concatenate(points_xyz_list, axis=0)
    colors = np.concatenate(colors, axis=0)
    marker_sizes = np.concatenate(marker_sizes, axis=0)
    fig = view_points(points_xyz, colors=colors, marker_size=marker_sizes, visible_axis=visible_axis)
    return fig


def view_pointcloud(pointcloud, marker_size=None, visible_axis=True):

    points = pointcloud[:, :3]
    colors = pointcloud[:, 3:]
    return view_points(points, colors=colors, marker_size=marker_size, visible_axis=visible_axis)


def view_mesh(mesh):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # pack all colors:
    try:
        vertices_colors = np.asarray(mesh.vertex_colors)
    except:
        vertices_colors = []
    if len(vertices_colors) > 0:
        colors = np.stack([
            vertices_colors[triangles[:, 0]],
            vertices_colors[triangles[:, 1]],
            vertices_colors[triangles[:, 2]]], axis=-1)
        colors = np.mean(colors, axis=-1)
    else:
        colors = np.random.uniform(0.,1., (3,))

    fig = go.Figure(go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
        facecolor=colors,
    ), layout=go.Layout(scene=dict(aspectmode='data')))
    # TODO: Add facecolor
    return fig


def view_gif(gif_path):
    im = IPythonImage(gif_path)
    return im


def view_trajectory_gif(data_path, camera_indx, traj_indx):
    gif_path = get_output_trajectory_gif_path(data_path, camera_name='camera_{}'.format(camera_indx),
                                              traj_indx=traj_indx)
    #     im = IPythonImage(gif_path)
    im = view_gif(gif_path)
    return im

