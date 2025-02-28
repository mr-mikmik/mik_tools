import numpy as np
import matplotlib.pyplot as plt
import pyrender
import trimesh
from typing import List, Tuple, Union
from tqdm import tqdm

from mik_tools import tr, pose_to_matrix
from .camera import Camera
from .rendering_tools import mask_vision_observations, scale_trimesh
from mik_tools.aux.package_utils import get_mesh_path


class VisionSceneRenderer(object):
    """
    This class is created for rendering vision observation of batches of object configurations.
    """

    def __init__(self, shape_id:str, vision_cameras:List[Camera], mesh_path=None, mesh_scale:float=1.0, verbose:bool=True, partial_rendering:bool=False):
        """
        :param shape_id (str):
        :param vision_cameras (List[Camera]): List of vision cameras to render the observations
        :param mesh_path:
        :param mesh_scale:
        :param verbose:
        :param partial_rendering:
        """
        self._shape_id = None
        self.vision_cameras = vision_cameras
        self.mesh_scale = mesh_scale
        self.verbose = verbose
        self.partial_rendering = partial_rendering
        self.mesh_path = mesh_path
        self.object_mesh = None
        self.scene = None
        self.render_engine = None
        self.object_node = None # Future work: Extend to multiple object nodes
        self.vision_camera_nodes = []
        self.vision_cameras: List[Camera] = vision_cameras # List of vision cameras -- Nv: Number of vision cameras
        self.shape_id = shape_id

    @property
    def shape_id(self):
        return self._shape_id

    @shape_id.setter
    def shape_id(self, value):
        current_shape_id = self._shape_id
        need_update = current_shape_id != value
        self._shape_id = value
        # update the scene with the new shape
        if need_update:
            self._load_object_mesh()
            self._init_scene()

    @property
    def object_name(self):
        return f'{self.shape_id}.ply'

    def render_poses(self, w_X_of:np.ndarray, as_mask:bool=False) -> List[np.ndarray]:
        """
        Render a set of object poses
        Args:
            w_X_of (np.ndarray): numpy array of shape (..., 4, 4) representing the object poses in the world frame to be rendered
            as_mask (bool): if True, we will return the observations as masks (i.e. 0s and 1s). If False, we return depth maps.
        Returns:
            vision_observations (list[np.ndarray]): List of Nv arrays of the vision depths renderings, one per vision camera, each of shape (..., 1, w_vi, h_vi)
        """
        vision_observations = self.render_vision(w_X_of, as_mask=as_mask)
        return vision_observations

    def render_vision(self, w_X_of:np.ndarray, as_mask:bool=False) -> List[np.ndarray]:
        """
        Render a set of object poses into vision observations
        Args:
            w_X_of (np.ndarray): numpy array of shape (..., 4, 4) representing the object poses in the world frame to be rendered
            as_mask (bool): if True, we will return the observations as masks (i.e. 0s and 1s). If False, we return depth maps.
        Returns:
            vision_observations (list[np.ndarray]): List of Nv arrays of the vision depths renderings, one per vision camera, each of shape (..., 1, w_vi, h_vi)
        """
        vision_observations = []
        for vision_indx in range(len(self.vision_cameras)):
            vision_depth_i = self._render_vision_indx(w_X_of, vision_indx)
            vision_observations.append(vision_depth_i)
        if as_mask:
            vision_observations = mask_vision_observations(vision_observations) # convert from depth maps to masks
        return vision_observations

    def render_vision_color(self, w_X_of:np.ndarray) -> List[np.ndarray]:
        """
        Render a set of object poses
        Args:
            w_X_of (np.ndarray): numpy array of shape (..., 4, 4) representing the object poses in the world frame to be rendered
        Returns:
            vision_observations (list[np.ndarray]): List of Nv arrays of the vision depths renderings, one per vision camera, each of shape (..., 3, w_vi, h_vi)
        """
        vision_observations = []
        for vision_indx in range(len(self.vision_cameras)):
            vision_color_i = self._render_vision_color_indx(w_X_of, vision_indx)
            vision_observations.append(vision_color_i)
        return vision_observations

    def _render_vision_indx(self, w_X_of:np.ndarray, vision_indx:int) -> np.ndarray:
        """
        Render a set of object poses for a given tactile camera
        Args:
            w_X_of (np.ndarray): numpy array of shape (..., 4, 4) representing the object poses in the world frame to be rendered
            vision_indx (int): tactile camera index to be rendered
        Returns:
            vision_depths (np.ndarray): vision depths rendered of shape (..., 1, w_vi, h_vi)
        """
        if self.scene is None:
            self._init_scene()
        vision_camera = self.vision_cameras[vision_indx]
        camera_node = self.vision_camera_nodes[vision_indx]
        depth_imgs = self._render_camera_object_poses(w_X_of, camera=vision_camera, camera_node=camera_node, only_depth=True)
        vision_depths = np.expand_dims(depth_imgs, axis=-3)
        return vision_depths

    def _render_vision_color_indx(self, w_X_of:np.ndarray, vision_indx:int, return_depth=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Render a set of object poses for a given tactile camera
        Args:
            w_X_of (np.ndarray): numpy array of shape (..., 4, 4) representing the object poses in the world frame to be rendered
            vision_indx (int): tactile camera index to be rendered
        Returns:
            vision_depths (np.ndarray): vision depths rendered of shape (..., 3, w_vi, h_vi)
            vision_depths (np.ndarray): vision depths rendered of shape (..., 1, w_vi, h_vi) -- only if return_depth=True
        """
        if self.scene is None:
            self._init_scene()
        vision_camera = self.vision_cameras[vision_indx]
        camera_node = self.vision_camera_nodes[vision_indx]
        vision_color, depth_imgs = self._render_camera_object_poses(w_X_of, camera=vision_camera, camera_node=camera_node)
        vision_color = np.einsum('...whc->...cwh', vision_color) # (..., 3, wv, hv)
        if return_depth:
            vision_depths = np.expand_dims(depth_imgs, axis=-3)
            return vision_color, vision_depths
        return vision_color

    def _render_camera_object_poses(self, w_X_of:np.ndarray, camera:Camera, camera_node, only_depth=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """

        Args:
            w_X_of (np.ndarray): numpy array of shape (..., 4, 4) representing the object poses in the world frame to be rendered
            camera (Camera):
            camera_node ():
        Returns:
            color_imgs (np.ndarray): of shape (..., 3, w, h)
            depth_imgs (np.ndarray): of shape (..., w, h)
        """
        # create the render engine
        if only_depth:
            depth_imgs = self._render_camera_object_poses_depth(w_X_of, camera, camera_node)
            return depth_imgs
        else:
            color_imgs, depth_imgs = self._render_camera_object_poses_rgb_depth(w_X_of, camera, camera_node)
            return color_imgs, depth_imgs

    def _render_camera_object_poses_rgb_depth(self, w_X_of: np.ndarray, camera: Camera, camera_node) -> Tuple[np.ndarray, np.ndarray]:
        """

        Args:
            w_X_of (np.ndarray): numpy array of shape (..., 4, 4) representing the object poses in the world frame to be rendered
            camera ():
            camera_node ():
        Returns:
            color_imgs (np.ndarray): of shape (..., w, h, 3)
            depth_imgs (np.ndarray): of shape (..., w, h)
        """
        if self.scene is None:
            self._init_scene()
        # create the render engine
        if self.partial_rendering:
            img_size = camera.render_img_size
            viewport_width, viewport_height = img_size[0], img_size[1]
        else:
            viewport_width,viewport_height = camera.img_size[0], camera.img_size[1]
        render_engine = pyrender.OffscreenRenderer(viewport_width=viewport_width,
                                                   viewport_height=viewport_height)
        # set the current camera as the active for the renderings
        self.scene.main_camera_node = camera_node
        color_imgs = []
        depth_imgs = []

        input_size = w_X_of.shape
        batch_dims = w_X_of.shape[:-2]  # this is the dimesnions of (...,)

        w_X_of = w_X_of.reshape(-1, *input_size[-2:])  # (B, 4, 4)

        frame_indxs = range(w_X_of.shape[0])
        if self.verbose:
            frame_indxs = tqdm(frame_indxs)
        for idx_frame in frame_indxs:
            self.scene.set_pose(self.object_node, pose=w_X_of[idx_frame])
            color, depth = render_engine.render(self.scene)

            color_imgs.append(color)
            depth_imgs.append(depth)
        render_engine.delete()
        # stack
        color_imgs = np.stack(color_imgs, axis=0) # (B, w, h, 3)
        depth_imgs = np.stack(depth_imgs, axis=0) # (B, w, h)
        # flip them on the width and height channels, also color to be rgb
        color_imgs = np.flip(color_imgs, axis=[-1, -2, -3]) # (B, w, h, 3)
        depth_imgs = np.flip(depth_imgs, axis=[-1, -2]) # (B, w, h)

        # reshape them
        color_imgs = color_imgs.reshape(*batch_dims, *color_imgs.shape[-3:])  # (..., w, h, 3)
        depth_imgs = depth_imgs.reshape(*batch_dims, *depth_imgs.shape[-2:])  # (..., w, h)

        return color_imgs, depth_imgs

    def _render_camera_object_poses_depth(self, w_X_of: np.ndarray, camera: Camera, camera_node) -> np.ndarray:
        """

        Args:
            w_X_of (np.ndarray): numpy array of shape (..., 4, 4) representing the object poses in the world frame to be rendered
            camera ():
            camera_node ():
        Returns:
            depth_imgs (np.ndarray): of shape (..., w, h)
        """
        if self.scene is None:
            self._init_scene()
        # create the render engine
        if self.partial_rendering:
            img_size = camera.render_img_size
            viewport_width, viewport_height = img_size[0], img_size[1]
        else:
            viewport_width, viewport_height = camera.img_size[0], camera.img_size[1]
        render_engine = pyrender.OffscreenRenderer(viewport_width=viewport_width,
                                                   viewport_height=viewport_height)
        # set the current camera as the active for the renderings
        self.scene.main_camera_node = camera_node
        depth_imgs = []

        input_size = w_X_of.shape
        batch_dims = w_X_of.shape[:-2] # this is the dimesnions of (...,)


        w_X_of = w_X_of.reshape(-1, *input_size[-2:]) # (B, 4, 4)

        frame_indxs = range(w_X_of.shape[0])
        if self.verbose:
            frame_indxs = tqdm(frame_indxs)
        for idx_frame in frame_indxs:
            self.scene.set_pose(self.object_node, pose=w_X_of[idx_frame])
            depth = render_engine.render(self.scene, flags=pyrender.RenderFlags.DEPTH_ONLY)

            depth_imgs.append(depth)
        render_engine.delete()
        # stack
        depth_imgs = np.stack(depth_imgs, axis=0) # (B, w, h)
        # flip them on the width and hight channels
        depth_imgs = np.flip(depth_imgs, axis=[-1, -2]) # (B, w, h)

        # reshape them
        depth_imgs = depth_imgs.reshape(*batch_dims, *depth_imgs.shape[-2:]) # (..., w, h)
        if self.partial_rendering:
            depth_imgs = camera.crop_rendering(depth_imgs)

        return depth_imgs

    def _load_object_mesh(self):
        if self.mesh_path is None:
            mesh_path = get_mesh_path(self.object_name)
        else:
            mesh_path = self.mesh_path
        object_trimesh = scale_trimesh(trimesh.load(mesh_path), scale=self.mesh_scale)
        self.object_mesh = pyrender.Mesh.from_trimesh(object_trimesh)

    def _init_scene(self):
        # create the scene
        self.scene = pyrender.Scene()

        # create the object node
        self.object_node = self.scene.add(self.object_mesh, pose=np.eye(4), name="obj")

        # Add the cameras
        self.scene, self.vision_camera_nodes = self._add_vision_cameras_to_scene(self.scene, return_cam_nodes=True)

    def _add_vision_cameras_to_scene(self, scene, return_cam_nodes=False):
        # put the cameras
        cf_X_cfpr = tr.euler_matrix(0., np.pi, 0)  # from camera pose to render pose
        cam_nodes = []
        for vcam in self.vision_cameras:
            if self.partial_rendering:
                K = vcam.K_rendering
            else:
                K = vcam.K
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]
            w_X_cfr = vcam.w_X_cf @ cf_X_cfpr  # adjust the axis rotation of pyrender
            cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
            vcam_node = scene.add(cam, pose=w_X_cfr)
            cam_nodes.append(vcam_node)
            # add light
            light = pyrender.SpotLight(color=np.ones(3), intensity=.2,
                                       innerConeAngle=np.pi / 16.0,
                                       outerConeAngle=np.pi / 6.0)
            scene.add(light, pose=w_X_cfr)
        if return_cam_nodes:
            out = scene, cam_nodes
        else:
            out = scene
        return out
