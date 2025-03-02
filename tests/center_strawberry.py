import numpy as np
import trimesh
import vedo


from mik_tools import matrix_to_pose, pose_to_matrix, transform_matrix_inverse, tr, eye_pose
from mik_tools.aux.package_utils import get_test_mesh_path
from mik_tools.visualization_tools.vedo_tools import draw_mesh, draw_vectors, draw_points, draw_frame, get_object_mesh
from mik_tools.data_utils.mesh_utils import discretize_mesh


mesh_path = get_test_mesh_path('strawberry.ply')
mesh = trimesh.load(mesh_path)

w_X_of = np.eye(4)

# get the center of the strawberry mesh
of_pos_of_new = -mesh.centroid
of_rpy_of_new = np.array([0., 0., 0.])
of_X_of_new = pose_to_matrix(np.concatenate([of_pos_of_new, of_rpy_of_new]))

# find the axis of rotation of the mesh
points = mesh.vertices
center = np.mean(points, axis=0)
points_centered = points - center
cov = points_centered.T @ points_centered
_, eigvecs = np.linalg.eig(cov)
axis = eigvecs[:, -1]
#
# # set the axis to the new z axis of the mesh
# axis = axis / np.linalg.norm(axis)
# z_axis = np.array([0., 0., 1.])
# angle = np.arccos(axis @ z_axis)
# rotation_axis = np.cross(axis, z_axis)
# rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
# rotation_matrix = tr.rotation_matrix(angle, rotation_axis)
# # of_X_of_new[:3, :3] = rotation_matrix[:3, :3]
# of_X_of_new = rotation_matrix


# move the mesh to the center
w_X_of_new = w_X_of @ of_X_of_new


view_mesh = get_object_mesh(mesh_path)


ax_scale = 25.
viz = vedo.Plotter(shape=(1, 3))
viz.at(0)
_ = draw_frame(pose=eye_pose, scale=ax_scale, viz=viz)
_ = draw_mesh(view_mesh.copy(), w_pose_of=matrix_to_pose(w_X_of), color='blue', alpha=.5, viz=viz)
# draw the axis
_ = draw_vectors(center[None,:], axis[None,:]*10, color='red', viz=viz)

viz.at(1)
_ = draw_frame(pose=eye_pose, scale=ax_scale, viz=viz)
_ = draw_mesh(view_mesh.copy(), w_pose_of=matrix_to_pose(of_X_of_new.copy()), color='green', alpha=.5, viz=viz)

viz.at(2)
_ = draw_frame(pose=eye_pose, scale=ax_scale, viz=viz)
_ = draw_mesh(view_mesh.copy(), w_pose_of=matrix_to_pose(w_X_of), color='blue', alpha=.5, viz=viz)
_ = draw_mesh(view_mesh.copy(), w_pose_of=matrix_to_pose(of_X_of_new), color='green', alpha=0.5, viz=viz)


viz.show().interactive()

# print the new pose of the mesh

