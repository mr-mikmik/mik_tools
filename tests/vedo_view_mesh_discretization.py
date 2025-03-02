import numpy as np
import trimesh
import vedo


from mik_tools import matrix_to_pose, pose_to_matrix, transform_matrix_inverse, tr, eye_pose
from mik_tools.aux.package_utils import get_test_mesh_path
from mik_tools.visualization_tools.vedo_tools import draw_mesh, draw_vectors, draw_points
from mik_tools.data_utils.mesh_utils import discretize_mesh


mesh_path = get_test_mesh_path('strawberry.ply')
mesh = trimesh.load(mesh_path)


# create a side-by-side visualization
viz = vedo.Plotter(shape=(1, 4))

viz.at(0)
_ = draw_mesh(mesh, w_pose_of=eye_pose, color=None, alpha=1.0, viz=viz)
# show

normal_scale = 0.1

mesh_simplified = mesh.simplify_quadric_decimation(face_count=1000)
points, normals, As = discretize_mesh(mesh_simplified)
viz.at(1)
viz = draw_mesh(mesh_simplified, w_pose_of=eye_pose, color=None, alpha=1.0, viz=viz)
# draw the points
viz = draw_points(points, color='red', viz=viz)
# draw normals
normals_As = normals * As[:, None] * normal_scale # (N, 3)
viz = draw_vectors(points, normals_As, color='blue', viz=viz)

mesh_simplified = mesh.simplify_quadric_decimation(face_count=100)
points, normals, As = discretize_mesh(mesh_simplified)
viz.at(2)
viz = draw_mesh(mesh_simplified, w_pose_of=eye_pose, color=None, alpha=1.0, viz=viz)
# draw the points
viz = draw_points(points, color='red', viz=viz)
# draw normals
normals_As = normals * As[:, None] * normal_scale # (N, 3)
viz = draw_vectors(points, normals_As, color='blue', viz=viz)

viz.at(3)
_ = draw_mesh(mesh, w_pose_of=eye_pose, color=None, alpha=1.0, viz=viz)
viz = draw_vectors(points, normals_As, color='blue', viz=viz)

viz.show().interactive()


