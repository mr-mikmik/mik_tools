import numpy as np
import matplotlib.pyplot as plt

from mik_tools import matrix_to_pose, pose_to_matrix, tr, transform_matrix_inverse, get_dataset_path
from mik_tools.camera_tools.cnos import CNOS
from mik_tools.camera_tools.camera_utils import compute_camera_pose
from mik_tools.rendering_tools import Camera
from mik_tools.rendering_tools import VisionSceneRenderer

shape_id = 'strawberry'
mesh_path = '/home/mik/robot_ws/src/megapose6d_ros/meshes/strawberry/nontextured.ply'

w_pose_cf = compute_camera_pose(np.array([.2, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
K = np.array([[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]])
camera = Camera(w_pose_cf)
vision_cameras = [camera]
vs_renderer = VisionSceneRenderer(shape_id=shape_id, vision_cameras=vision_cameras, mesh_path=mesh_path, mesh_scale=0.001)

# render the scene
w_X_of = np.eye(4)
color_imgs = vs_renderer.render_vision_color(w_X_of) # (1, 3, w, h)
depth_imgs = vs_renderer.render_vision(w_X_of) # (1, 1, w, h)

color_img_rgb = color_imgs[0].transpose(1, 2, 0) # (h, w, 3)
depth_img = depth_imgs[0][0] # (h, w)

fig, axes = plt.subplots(1,2)
axes[0].imshow(color_img_rgb)
axes[1].imshow(depth_img)
plt.show()

