import numpy as np
import torch

from mik_tools import matrix_to_pose, pose_to_matrix, transform_matrix_inverse, transform_points_3d, transform_vectors_3d, tr, eye_pose
from mik_tools.kinematic_tools import get_contact_jacobian, transform_wrench
from mik_tools.visualization_tools.vedo_tools.vis_3d_tools import draw_frame, draw_force, draw_torque, draw_wrench


# CASE 1:
def compute_jacobian_and_contact_wrench(w_X_cf, fc, w_X_of=None, visualize=False):
    if w_X_of is None:
        w_X_of = pose_to_matrix(eye_pose)
    of_X_w = transform_matrix_inverse(w_X_of)
    wf_X_of = pose_to_matrix(eye_pose)
    wf_X_cf = wf_X_of @ of_X_w @ w_X_cf

    Jc = get_contact_jacobian(wf_X_cf=wf_X_cf)
    contact_wrench_wf = Jc @ fc


    # visualize
    if visualize:
        print("contact_wrench_wf: ", contact_wrench_wf)
        viz = draw_frame(pose=matrix_to_pose(w_X_of), scale=1.0)
        viz = draw_frame(pose=matrix_to_pose(wf_X_cf), scale=.5, viz=viz)
        fc_w = transform_vectors_3d(fc, w_X_cf)
        viz = draw_force(fc_w, w_X_cf[:3, 3], viz=viz, color='pink')
        contact_wrench_w = transform_wrench(contact_wrench_wf, of_X_w)
        viz = draw_wrench(contact_wrench_w, w_X_of[:3, 3], viz=viz)
        viz.show()
    return Jc, contact_wrench_wf


w_pos_cf = np.array([0., 0., -1.])
w_quat_cf = tr.quaternion_from_euler(0., -np.pi*0.5, 0)
w_pose_cf = np.concatenate([w_pos_cf, w_quat_cf])
w_X_cf = pose_to_matrix(w_pose_cf)
fc = np.array([1., 0, 0.])  # pure normal force
Jc, contact_wrench_wf = compute_jacobian_and_contact_wrench(w_X_cf, fc, visualize=True)


w_pos_cf = np.array([1., 0., -1.])
w_quat_cf = tr.quaternion_from_euler(0., -np.pi*0.5, 0)
w_pose_cf = np.concatenate([w_pos_cf, w_quat_cf])
w_X_cf = pose_to_matrix(w_pose_cf)
fc = np.array([1., 0, 0.])  # pure normal force
Jc, contact_wrench_wf = compute_jacobian_and_contact_wrench(w_X_cf, fc, visualize=True)


w_pos_cf = np.array([1., 1., -1.])
w_quat_cf = tr.quaternion_from_euler(0., -np.pi*0.5, 0)
w_pose_cf = np.concatenate([w_pos_cf, w_quat_cf])
w_X_cf = pose_to_matrix(w_pose_cf)
fc = np.array([1., 0, 0.])  # pure normal force
Jc, contact_wrench_wf = compute_jacobian_and_contact_wrench(w_X_cf, fc, visualize=True)

