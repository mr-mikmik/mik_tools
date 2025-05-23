import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
import os

from mik_tools.aux.package_utils import get_test_mesh_path

if 'WAYLAND_DISPLAY' in os.environ and 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
if 'WAYLAND_DISPLAY' in os.environ:
    del os.environ['WAYLAND_DISPLAY']

# mesh_name = 'fuze.obj'
mesh_name = 'drill.obj'

mesh_path = get_test_mesh_path(mesh_name)
fuze_trimesh = trimesh.load(mesh_path)
mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
scene = pyrender.Scene()
scene.add(mesh)

camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
s = np.sqrt(2)/2
camera_pose = np.array([
   [0.0, -s,   s,   0.3],
   [1.0,  0.0, 0.0, 0.0],
   [0.0,  s,   s,   0.35],
   [0.0,  0.0, 0.0, 1.0],
])
scene.add(camera, pose=camera_pose)
light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                           innerConeAngle=np.pi/16.0,
                           outerConeAngle=np.pi/6.0)
scene.add(light, pose=camera_pose)
r = pyrender.OffscreenRenderer(400, 400)
color, depth = r.render(scene)
plt.figure()
plt.subplot(1,2,1)
plt.axis('off')
plt.imshow(color)
plt.subplot(1,2,2)
plt.axis('off')
plt.imshow(depth, cmap=plt.cm.gray_r)
plt.show()
