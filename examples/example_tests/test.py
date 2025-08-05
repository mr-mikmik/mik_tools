import numpy as np
import matplotlib.pyplot as plt

from mik_tools import matrix_to_pose, pose_to_matrix, tr, transform_matrix_inverse, get_dataset_path
from mik_tools.camera_tools.cnos import CNOS
from mik_tools.camera_tools.camera_utils import compute_camera_pose
from mik_tools.rendering_tools import Camera
from mik_tools.rendering_tools import VisionSceneRenderer


# Try the same but with mik cube:

shape_id = 'rubik_block'
mesh_path = '/home/mik/robot_ws/src/megapose6d_ros/meshes/rubik_block/rubik_block_texture.obj'

w_pose_cf = compute_camera_pose(np.array([.2, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
K = np.array([[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]])
camera = Camera(w_pose_cf)
vision_cameras = [camera]

vs_renderer = VisionSceneRenderer(shape_id=shape_id, vision_cameras=vision_cameras, mesh_path=mesh_path, mesh_scale=0.001)
