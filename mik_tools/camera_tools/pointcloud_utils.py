import os

import numpy as np
try:
    # check if is macos
    import sys
    if not sys.platform == 'darwin':
        import open3d as o3d
        from open3d.visualization import Visualizer
except (ModuleNotFoundError, ImportError):
    pass
import torch
from scipy.spatial import KDTree

from mik_tools.mik_tf_tools import transformations as tr


EPS = 1e-9


def save_pointcloud(pc, filename, save_path):
    """
    Save the array of points provided as a .ply file
    :param pc: Point cloud as an array (N,6), where last dim is as:
        - X Y Z R G B
    :param name:
    :return: None
    """
    num_points = pc.shape[0]
    point_lines = []
    pc_color = pc[:, 3:]
    if np.all(pc_color <= 1):
        pc[:, 3:] *= 255
    for point in pc:
        point_lines.append(
            "{:f} {:f} {:f} {:d} {:d} {:d} 255\n".format(point[0], point[1], point[2], int(point[3]), int(point[4]),
                                                         int(point[5])))
    points_text = "".join(point_lines)
    file_name = '{}.ply'.format(filename)
    pc_path = os.path.join(save_path, file_name)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    with open(pc_path, 'w+') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex {}\n'.format(num_points))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('property uchar alpha\n')
        f.write('end_header\n')
        f.write(points_text)

    # print('PC saved as {}'.format(pc_path))


def load_pointcloud(pc_path, as_array=False):
    """
    Load a pointcloud as a (N,6) array
    Args:
        pc_path: path to the file containint the pointcloud

    Returns: <np.ndarray> of size (N,6)
    """
    pcd = o3d.io.read_point_cloud(pc_path)
    pc_out = pcd
    if as_array:
        pc_out = unpack_o3d_pcd(pcd)
    return pc_out


def pack_o3d_pcd(pc_array):
    """
    Given a pointcloud as an array (N,6), convert it to open3d PointCloud
    Args:
        pc_array: <np.ndarray> of size (N,6) containing the point in x y z r g b

    Returns: o3d.PointCloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_array[:, :3])
    if pc_array.shape[-1] > 3:
        pcd.colors = o3d.utility.Vector3dVector(pc_array[:, 3:7])
    return pcd


def unpack_o3d_pcd(pcd):
    """
    Convert an open3d PointCloud into a numpy array of size (N, 6)
    Args:
        pcd: open3d PointCloud object
    Returns: <np.ndarray> of size (N, 6)
    """
    pc_xyz = np.asarray(pcd.points)
    pc_rgb = np.asarray(pcd.colors)
    if len(pc_rgb) <= 0:
        pc_rgb = np.zeros_like(pc_xyz)
    pc_array = np.concatenate([pc_xyz, pc_rgb], axis=-1)
    return pc_array


def mesh_to_pointcloud(mesh, num_points=1000):
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    pointcloud = unpack_o3d_pcd(pcd)
    return pointcloud


def view_pointcloud(pc, frame=False, scale=1., addital_geometries_to_view=None):
    """
    Simple visualization of pointclouds
    Args:
        pc: pointcloud array or a list of pointcloud arrays

    Returns:

    """
    pcds = []
    if type(pc) is not list:
        pc = [pc]
    for pc_i in pc:
        if not isinstance(pc_i, o3d.geometry.PointCloud):
            pcd_i = pack_o3d_pcd(pc_i)
        else:
            pcd_i = pc_i
        pcds.append(pcd_i)
    view_pcd(pcds, frame=frame, scale=scale, addital_geometries_to_view=addital_geometries_to_view)


def view_pcd(pcds, frame=False, scale=1.0, addital_geometries_to_view=None):
    if type(pcds) is not list:
        pcds = [pcds]
    first_pcd = pcds[0]
    first_points = np.asarray(first_pcd.points)
    if frame:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale * 0.5 * np.std(first_points),
                                                                       origin=[0, 0, 0])
        pcds.append(mesh_frame)
    if addital_geometries_to_view is not None:
        pcds = pcds + addital_geometries_to_view
    o3d.visualization.draw_geometries(pcds)


def get_far_points_indxs(ref_pc, query_pc, d_threshold):
    """
    Compare the query_pc with the ref_pc and return the points in query_pc that are farther than d_threshold from ref_pc
    Args:
        ref_pc: <np.ndarray> (N,6)  reference point cloud
        query_pc: <np.ndarray> (N,6) query point cloud,
        d_threshold: <float> threshold distance to consider far if d>d_threshold
    Returns:
        - list of points indxs in query_pc that are far from ref_pc
    """
    ref_xyz = ref_pc[:, :3]
    qry_xyz = query_pc[:, :3]
    ref_tree = KDTree(ref_xyz)
    near_qry_indxs = ref_tree.query_ball_point(qry_xyz, d_threshold)
    far_qry_indxs = [i for i, x in enumerate(near_qry_indxs) if len(x) == 0]
    return far_qry_indxs


def get_far_points(ref_pc, query_pc, d_threshold):
    """
    Compare the query_pc with the ref_pc and return the points in query_pc that are farther than d_threshold from ref_pc
    Args:
        ref_pc: <np.ndarray> (N,6)  reference point cloud
        query_pc: <np.ndarray> (N,6) query point cloud,
        d_threshold: <float> threshold distance to consider far if d>d_threshold
    Returns:
        - <np.ndarray> (n,6) containing the points in query_pc that are far from ref_pc. n<=N
    """
    far_qry_indxs = get_far_points_indxs(ref_pc, query_pc, d_threshold)
    far_qry_points = query_pc[far_qry_indxs]
    return far_qry_points


def tr_pointcloud(pc, R=None, t=None, T=None):
    """
    Transform a point cloud given the homogeneous transformation represented by R,t
    R and t can be seen as a tranformation new_frame_X_old_frame where pc is in the old_frame and we want it to be in the new_frame
    Args:
        pc: (..., N, 6) pointcloud or (..., N, 3) pointcloud
        R: (3,3) numpy array i.e. target_R_source
        t: (3,) numpy array i.e. target_t_source
        T: (4,4) numpy array representing the tranformation i.e. target_T_source
    Returns:
        pc_tr: (..., N, 6) pointcloud or (..., N, 3) pointcloud transformed in the
    """
    if T is not None:
        R = T[:3, :3]
        t = T[:3, 3]
    pc_xyz = pc[..., :3]
    # pc_xyz_tr = pc_xyz@R.T + t
    pc_xyz_tr = np.einsum('ij,...lj->...li', R, pc_xyz) + t

    # handle RGB info held in the other columns
    if pc.shape[-1] > 3:
        pc_rgb = pc[..., 3:7]
        pc_tr = np.concatenate([pc_xyz_tr, pc_rgb], axis=-1)
    else:
        pc_tr = pc_xyz_tr
    return pc_tr


def find_best_transform(point_cloud_A, point_cloud_B):
    """
    Find the transformation 2 corresponded point clouds.
    Note 1: We assume that each point in the point_cloud_A is corresponded to the point in point_cloud_B at the same location.
        i.e. point_cloud_A[i] is corresponded to point_cloud_B[i] forall 0<=i<N
    :param point_cloud_A: np.array of size (N, 6) (scene)
    :param point_cloud_B: np.array of size (N, 6) (model)
    :return:
         - t: np.array of size (3,) representing a translation between point_cloud_A and point_cloud_B
         - R: np.array of size (3,3) representing a 3D rotation between point_cloud_A and point_cloud_B
    Note 2: We transform the model to match the scene.
    """
    pcs_A = point_cloud_A[:, :3]
    pcs_B = point_cloud_B[:, :3]
    mu_a = np.mean(pcs_A, axis=0)
    mu_b = np.mean(pcs_B, axis=0)
    a = pcs_A - mu_a
    b = pcs_B - mu_b
    W = a.T @ b
    U, S, Vh = np.linalg.svd(W)
    R_star = U @ Vh
    if np.linalg.det(R_star) < 0:
        # Fix improper rotation
        Vh[-1, :] *= -1
        R_star = U @ Vh # det(R_star) = +1
    t_star = mu_a - R_star@ mu_b

    t = t_star
    R = R_star
    return t, R


def get_projection_tr(projection_axis):
    projection_axis = projection_axis/np.linalg.norm(projection_axis)
    z_axis = np.array([0, 0, 1])
    if np.all(projection_axis == z_axis):
        projection_tr = np.eye(4)
    else:
        rot_angle = np.arccos(np.dot(projection_axis, z_axis))
        _rot_axis = np.cross(z_axis, projection_axis)
        rot_axis = _rot_axis/np.linalg.norm(_rot_axis)
        projection_tr = tr.quaternion_matrix(tr.quaternion_about_axis(rot_angle, axis=rot_axis))
    return projection_tr


def project_pc(pc, projection_axis):
    # pc: numpy array or tensor of shape (..., num_points, num_dims)
    # projection_axis (num_dims, ) array
    pc_shape = pc.shape
    is_batched = len(pc_shape) == 2
    is_tensor = type(pc) is torch.Tensor
    projection_tr = get_projection_tr(projection_axis)  # (4, 4) transformation matrix
    if is_tensor:
        # convert projection_tr to tensor
        projection_tr = torch.tensor(projection_tr).to(pc.device)
    if not is_batched:
        pc = pc.reshape((-1,) + pc_shape[-1:]) # add a batch dimension first, or combine the other dimesions into one

    R = projection_tr[:3, :3] # size (3,3)
    t = projection_tr[:3, 3] # (3,)
    if is_tensor:
        pc_rot = torch.einsum('ij,lj->li', R, pc)  # (n_points_total, n_coords)
    else:
        pc_rot = np.einsum('ij,lj->li', R, pc) # (n_points_total, n_coords)
    pc_tr = pc_rot + t  # (n_points_total, n_coords)
    projected_pc = pc_tr.reshape(pc_shape)
    return projected_pc


def generate_partial_view(mesh, view_axis, up_axis=np.array([0, 0, 1]), z_view_angle=0, look_at=np.array([0., 0., 0.]), view=False):
    """
    Generate a partial view from the mesh
    :param mesh: open3d mesh
    :param view_axis: np.array of shape (3,) that defines the view axis (by default the camera
    :param z_view_angle:
    :return:
    """

    vis = Visualizer()
    vis.create_window(visible=view)
    vis.add_geometry(mesh)
    # set visualization parameters to veiw from the x axis

    if 1 - abs(np.dot(up_axis, view_axis)) < EPS:
        view_axis = up_axis.copy()
        R = tr.quaternion_matrix(tr.quaternion_about_axis(angle=z_view_angle, axis=view_axis))[:3, :3]
        up_axis = R @ np.array([1., 0., 0.])

    vis.update_renderer()
    # Generate a view from the scene
    ctr = vis.get_view_control()
    # import pdb; pdb.set_trace()
    # ctr.change_field_of_view(step=-90)
    ctr.set_lookat(look_at)
    # ctr.rotate(10., 0.)
    # put the camera to the (1,0,0) position
    ctr.set_up(up_axis)
    ctr.set_front(view_axis)
    # ctr.set_zoom(1.0)
    # set the camera resolution
    # ctr.set_constant_z_far(0.1)
    # ctr.set_constant_z_near(0.01)
    camera = ctr.convert_to_pinhole_camera_parameters()
    vis.update_renderer()
    # set the camera control to the visualizer
    if view:
        vis.run()
    vis.update_renderer()
    # capture the depth buffer
    depth_img = vis.capture_depth_float_buffer(do_render=True)
    vis.destroy_window()
    # close the visualizer
    vis.close()
    return depth_img, camera


def generate_partial_pc(*args, **kwargs):
    depth_img, camera = generate_partial_view(*args, **kwargs)
    # transform depth_raw to open3d iamge
    # transform depth to a pointcloud
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_img,intrinsic=camera.intrinsic, extrinsic=camera.extrinsic)
    return pcd



# TEST:
if __name__ == '__main__':
    TEST_PATH = '/home/mik/Desktop/test'
    YCB_MUG_PATH = '/home/mik/Downloads/025_mug/google_16k/nontextured.ply'

    mesh = o3d.io.read_triangle_mesh(YCB_MUG_PATH)

    # view mesh
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([mesh, frame])

    view_axis = np.array([-1., 0, 0])
    # view_axis = np.array([.0, 0, -1.])
    pcd = generate_partial_pc(mesh, view_axis, look_at=np.array([0., 0., 0.05]), view=False)
    o3d.visualization.draw_geometries([pcd, frame])
    # further crop the pointcloud to get only the points that are close to the maximum in x
    x_min = np.min(pcd.points, axis=0)[0]
    max_d = 0.02
    pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox([-np.inf, -np.inf, -np.inf], [x_min + max_d, np.inf, np.inf]))
    # view the pointcloud
    o3d.visualization.draw_geometries([pcd, frame])

