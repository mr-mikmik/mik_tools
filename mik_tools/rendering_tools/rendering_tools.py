import numpy as np
import matplotlib.pyplot as plt
import pyrender
import cv2
import copy
from trimesh import Trimesh
from typing import List, Union, Tuple
import torch

from mik_tools import tr
from mik_tools.camera_tools.pointcloud_utils import tr_pointcloud
from mik_tools.camera_tools.img_utils import project_depth_image
from mik_tools.mik_learning_tools import batched_img_decorator

# os.environ['PYOPENGL_PLATFORM'] = 'egl'
# OBSERVATION: with egl, we need to run scripts with EGL_DEVICE_ID=-1 bash xxx.sh, or add export EGL_DEVICE_ID=-1


EPS = 0.0001
DEPTH_CUTOFF = 1.
TACTILE_PLANE_DIST = 36


def create_intrinsic_matrix(fx, fy, cx, cy):
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return K


def scale_tr(t1_X_t2, scale=1.0):
    t1_X_t2_scaled = t1_X_t2.copy()
    t1_X_t2_scaled[:3,3] *= scale
    return t1_X_t2_scaled


def scale_trimesh(object_trimesh:Trimesh, scale=1.0):
    scale_matrix = np.eye(4)
    scale_matrix[:3, :3] *= scale
    object_trimesh_scaled = copy.deepcopy(object_trimesh).apply_transform(scale_matrix)
    return object_trimesh_scaled


def scale_object_mesh(object_trimesh:Trimesh, scale=1.0):
    object_trimesh_scaled = scale_trimesh(object_trimesh, scale=scale)
    object_mesh_scaled = pyrender.Mesh.from_trimesh(object_trimesh_scaled)
    return object_mesh_scaled


def get_default_intrinsics():
    fx = 700.
    fy = 700.
    cx = 200.
    cy = 200.
    K = create_intrinsic_matrix(fx, fy, cx, cy)
    return K


def render_view(object_mesh: pyrender.Mesh, of_X_cf:np.ndarray, K=None, img_size=(400,400), view: bool=False, blur=False):
    """
    Render a view from the object mesh placing the camera at the relative location of_X_cf w.r.t to the mesh frmae
    Args:
        object_mesh (pyrender.Mesh):
        of_X_cf (): (4,4) array representing the camera pose w.r.t the object mesh frame.
        K (np.ndarray): (3,3) array representing the camera intrinsic matrix
        view (bool),:

    Returns:
        color, depth
    NOTE: pyrender renders a camera that has the axis as -z. However, we have the camera view as +z
    Need to fix this by setting a transformation

    """

    if K is None:
        K = get_default_intrinsics()

    cf_X_cfpr = tr.euler_matrix(0., np.pi, 0)  # from camera pose to render pose

    w_X_of = np.eye(4)  # set world frame the same as object frmae
    w_X_cf = w_X_of @ of_X_cf @ cf_X_cfpr

    #     camera = pyrender.PerspectiveCamera(yfov=np.pi / 5.0, aspectRatio=1.0)
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]

    camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy)

    scene = pyrender.Scene()
    scene.add(object_mesh)
    scene.add(camera, pose=w_X_cf)

    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                               innerConeAngle=np.pi / 16.0,
                               outerConeAngle=np.pi / 6.0)
    scene.add(light, pose=w_X_cf)
    r = pyrender.OffscreenRenderer(viewport_width=img_size[0], viewport_height=img_size[1])
    color, depth = r.render(scene)
    r.delete()

    # flip the axis
    depth = np.flip(depth)
    color = np.flip(color)

    if blur:
        depth_blur = cv2.medianBlur((depth).astype(np.uint8), 25)
        depth_blur = cv2.bilateralFilter(depth_blur, 100, 15, 15)
        depth = depth_blur.astype(np.float32)

    if view:
        fig, axes = plt.subplots(1, 2)
        axes[0].axis('off')
        axes[0].imshow(color)
        axes[1].axis('off')
        axes[1].imshow(depth, cmap=plt.cm.gray_r)
    return color, depth


def render_tactile(object_mesh, of_X_cf, blur=False, K=None, img_size=(400,400), tactile_plane_distance=None):
    if tactile_plane_distance is None:
        tactile_plane_distance = TACTILE_PLANE_DIST
    color, depth = render_view(object_mesh, of_X_cf, K=K, img_size=img_size, view=False)
    background_mask = depth > EPS
#     mask = background_mask & (depth < closest_depth + depth_cutoff)
    mask = background_mask & (depth < tactile_plane_distance)
    masked_depth = mask * depth
    if blur:
        masked_depth_blur = cv2.medianBlur((masked_depth).astype(np.uint8), 25)
        masked_depth_blur = cv2.bilateralFilter(masked_depth_blur, 100, 15, 15)
        tactile_depth = masked_depth_blur.astype(np.float32)
    else:
        tactile_depth = masked_depth
    return tactile_depth


def get_free_points(depth_img, K, min_depth=0., max_depth=1., num_steps=10, w_X_c=None):
    # create a grid of depths to be evaluates
    h,w = depth_img.shape[0], depth_img.shape[1]
    depth_values = np.linspace(min_depth, max_depth, num_steps+1)[1:]
    all_depths = np.repeat(np.expand_dims(depth_values, axis=1), w*h, axis=1).reshape(num_steps, h,w) # (num_steps, h, w)
    all_Ks = np.repeat(np.expand_dims(K, axis=0), num_steps, axis=0)
    # project the depths
    img_xyzs = project_depth_image(all_depths, all_Ks) # (num_steps, h, w, 3)
    depths = img_xyzs[..., -1] # (num_steps, h, w)
    # mask dpeths that z is closer than the depths img value
    depth_mask = depth_img.copy()
    depth_mask[np.where(depth_img == 0.0)] = np.inf
    mask = depths < depth_mask[None, :, :]
    pc_out = img_xyzs[mask] # (N, 3) where N<=num_steps*h*w
    if w_X_c is not None:
        pc_out = tr_pointcloud(pc_out, T=w_X_c)
    return pc_out


def subsample_points(points, num_points=1000):
    random_indices = np.random.choice(len(points), size=num_points, replace=False)
    points_subsampled = points[random_indices]
    return points_subsampled


def mask_tactile_observations(tactile_obs: List[Union[np.ndarray, torch.Tensor]]) -> List[Union[np.ndarray, torch.Tensor]]:
    """
    Mask the tactile observations from a depth map to a mask
    Args:
        tactile_obs (List[torch.Tensor]) containing Nt tensors as Nt x (..., 1, wv, hv)
    Returns:
        masked_tactile_obs (List[torch.Tensor]) containing Nt tensors as Nt x (..., 1, wv, hv) as float32
    """
    masked_tactile_obs = []
    for tobs in tactile_obs:
        if torch.is_tensor(tobs):
            masked_tobs = (tobs > EPS).to(torch.float32)
        else:
            masked_tobs = (tobs > EPS).astype(np.float32)
        masked_tactile_obs.append(masked_tobs)
    return masked_tactile_obs


def mask_vision_observations(vision_obs: List[Union[np.ndarray, torch.Tensor]]) -> List[Union[np.ndarray, torch.Tensor]]:
    masked_vision_obs = []
    for vobs in vision_obs:
        if torch.is_tensor(vobs):
            masked_vobs = (vobs > EPS).to(torch.float32)
        else:
            masked_vobs = (vobs > EPS).astype(np.float32)
        masked_vision_obs.append(masked_vobs)
    return masked_vision_obs


def robot_scene_masking(vision_observation:List[np.ndarray], robot_scene_observation:List[np.ndarray], return_masks=False) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[np.ndarray]]]:
    """
    Mask the vision obserations removing the parts that are not seen due to robot robot_scene_observation.
    Args:
        vision_observation (list[np.ndarray]): List of Nv arrays of the vision depths renderings of the object (without scene), one per vision camera, each of shape (..., 1, w_vi, h_vi)
        robot_scene_observation (list[np.ndarray]): List of Nv arrays of the vision depths renderings with only the robot (NOT Object!!!), one per vision camera, each of shape (1, w_vi, h_vi)
    Returns:
        masked_vision_observation (np.ndarray): of shape (K, Nv, 1, wv, hv) of the observable part of the object.
    """
    masked_vision_observation = []
    masks = []
    for vo, rso in zip(vision_observation, robot_scene_observation):
        # vo shape (..., 1, wv, hv)
        # rso shape (1, wv, hv)
        rso = np.expand_dims(rso, axis=0)  # (1, 1, wv, hv)
        @batched_img_decorator
        def batched_masking(vo_batched):
            # vo_batched shape (K, 1, wv, hv)
            # rso shape (1, wv, hv)
            # rso[rso<EPS] = 10000.
            obj_mask = vo > EPS # (K, 1, wv, hv) # TODO: Fix this so we do not get the zeros from the background interfere
            # front_mask = vo < rso  # (K, 1, wv, hv)
            front_mask = (vo_batched < rso) | (rso < EPS)  # (K, 1, wv, hv)
            mask = obj_mask * front_mask # (K, 1, wv, hv)
            masked_vo = vo * mask # (K, 1, wv, hv)
            out = np.concatenate([masked_vo, mask], axis=-3) # (K, 2, wv, hv)
            return out
        out = batched_masking(vo) # (..., 2, wv, hv)
        masked_vo = out[..., 0:1, :,:] # (..., 1, wv, hv)
        mask = out[..., 1:2, :,:] # (..., 1, wv, hv)
        masked_vision_observation.append(masked_vo)
        masks.append(mask)
    if return_masks:
        return masked_vision_observation, masks
    return masked_vision_observation


def batched_robot_scene_masking(vision_observation:np.ndarray, robot_scene_observation:np.ndarray) -> np.ndarray:
    """
    Mask the vision obserations removing the parts that are not seen due to robot robot_scene_observation.
    Args:
        vision_observation (np.ndarray): of shape (K, Nv, 1, wv, hv) of only the object
        robot_scene_observation (np.ndarray): of shape (Nv, 1, wv, hv) with only the robot (NOT Object!!!)
    Returns:
        masked_vision_observation (np.ndarray): of shape (K, Nv, 1, wv, hv) of the observable part of the object.
    """
    robot_scene_observation = np.expand_dims(robot_scene_observation, axis=0) # (1, Nv, 1, wv, hv)
    obj_mask = vision_observation > EPS # (K, Nv, 1, wv, hv)
    front_mask = vision_observation < robot_scene_observation # (K, Nv, 1, wv, hv)
    mask = obj_mask * front_mask # (K, Nv, 1, wv, hv)
    masked_vision_observation = vision_observation * mask # (K, Nv, 1, wv, hv)
    return masked_vision_observation
