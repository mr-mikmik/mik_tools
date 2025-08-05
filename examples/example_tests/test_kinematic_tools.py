import numpy as np

from mik_tools import tr, pose_to_matrix
from mik_tools.mik_tf_tools.tr_trajectory_utils import get_w_T_plane
from mik_tools.kinematic_tools import project_wrench_to_plane, project_wrench_to_plane_2
from mik_tools.kinematic_tools import vector_to_skew_matrix, get_adjoint_matrix, transform_twist, transform_wrench, twist_to_transform, get_adjoint_matrix_planar, transform_wrench_planar, project_wrench_to_plane, \
    project_wrench_to_plane_2, unproject_wrench_to_plane


# TESTS AND DEBUG:
if __name__ == '__main__':

    # # TEST WITHOUT ROTATIONS ------------------------
    # # test pose transfomations
    # plane = np.array([0, 0, 1])
    # point = np.array([1, 1, 0])
    # w_T_plane = get_w_T_plane(plane=plane, point=point)
    # w_T_wf = pose_to_matrix(np.array([4, 3, 1, 0., 0, 0, 1]))
    # plane_T_wf = np.linalg.inv(w_T_plane) @ w_T_wf
    # wf_T_plane = np.linalg.inv(plane_T_wf)
    # print("w_T_plane: ", w_T_plane)
    # print("plane_T_wf: ", plane_T_wf)
    #
    # # CASE 1 -
    # print('CASE 1')
    # wrench_wf = np.array([1, 0, 0, 0, 0, 0])
    # wrench_pf = transform_wrench(wrench_wf, wf_T_tf=wf_T_plane)
    # print("wrench_wf: ", wrench_wf)
    # print("wrench_pf: ", wrench_pf)
    # projected_wrench_wf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf)
    # projected_wrench_pf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf, plane_frame_coords=True)
    # print("projected_wrench_pf: ", projected_wrench_pf)
    # print("projected_wrench_wf: ", projected_wrench_wf)
    # projected_wrench_wf_2 = project_wrench_to_plane_2(wrench_wf, w_T_plane, w_T_wf)
    # print("projected_wrench_wf_2: ", projected_wrench_wf_2)
    # # CASE 1 -
    # print('CASE 2')
    # wrench_wf = np.array([1, 2, 3, 0, 0, 0])
    # wrench_pf = transform_wrench(wrench_wf, wf_T_tf=wf_T_plane)
    # print("wrench_wf: ", wrench_wf)
    # print("wrench_pf: ", wrench_pf)
    # projected_wrench_wf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf)
    # projected_wrench_pf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf, plane_frame_coords=True)
    # print("projected_wrench_pf: ", projected_wrench_pf)
    # print("projected_wrench_wf: ", projected_wrench_wf)
    # projected_wrench_wf_2 = project_wrench_to_plane_2(wrench_wf, w_T_plane, w_T_wf)
    # print("projected_wrench_wf_2: ", projected_wrench_wf_2)
    # # CASE 3 -
    # print('CASE 3')
    # wrench_wf = np.array([6, 5, 4, 3, 2, 1])
    # wrench_pf = transform_wrench(wrench_wf, wf_T_tf=wf_T_plane)
    # print("wrench_wf: ", wrench_wf)
    # print("wrench_pf: ", wrench_pf)
    # projected_wrench_wf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf)
    # projected_wrench_pf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf, plane_frame_coords=True)
    # print("projected_wrench_pf: ", projected_wrench_pf)
    # print("projected_wrench_wf: ", projected_wrench_wf)
    # projected_wrench_wf_2 = project_wrench_to_plane_2(wrench_wf, w_T_plane, w_T_wf)
    # print("projected_wrench_wf_2: ", projected_wrench_wf_2)

    # TEST ROTATIONS ------------------------
    # CASE 4 - --- Test with rotations on z axis
    plane = np.array([0, 0, 1])
    point = np.array([1, 1, 0])
    w_T_plane = get_w_T_plane(plane=plane, point=point)
    w_T_wf = pose_to_matrix(np.concatenate([np.array([4, 3, 1]), tr.quaternion_about_axis(np.pi/2, [0, 0, 1])])) # 90 degree rotation
    plane_T_wf = np.linalg.inv(w_T_plane) @ w_T_wf
    wf_T_plane = np.linalg.inv(plane_T_wf)
    print("w_T_plane: ", w_T_plane)
    print("plane_T_wf: ", plane_T_wf)
    print('CASE 4')
    wrench_wf = np.array([1, 0, 0, 0, 0, 0])
    wrench_pf = transform_wrench(wrench_wf, wf_T_tf=wf_T_plane)
    print("wrench_wf: ", wrench_wf)
    print("wrench_pf: ", wrench_pf)
    projected_wrench_wf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf)
    projected_wrench_pf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf, plane_frame_coords=True)
    print("projected_wrench_pf: ", projected_wrench_pf)
    print("projected_wrench_wf: ", projected_wrench_wf)
    projected_wrench_wf_2 = project_wrench_to_plane_2(wrench_wf, w_T_plane, w_T_wf)
    print("projected_wrench_wf_2: ", projected_wrench_wf_2)
    print('CASE 5')
    wrench_wf = np.array([0, 1, 0, 0, 0, 0])
    wrench_pf = transform_wrench(wrench_wf, wf_T_tf=wf_T_plane)
    print("wrench_wf: ", wrench_wf)
    print("wrench_pf: ", wrench_pf)
    projected_wrench_wf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf)
    projected_wrench_pf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf, plane_frame_coords=True)
    print("projected_wrench_pf: ", projected_wrench_pf)
    print("projected_wrench_wf: ", projected_wrench_wf)
    projected_wrench_wf_2 = project_wrench_to_plane_2(wrench_wf, w_T_plane, w_T_wf)
    print("projected_wrench_wf_2: ", projected_wrench_wf_2)
    print('CASE 6')
    wrench_wf = np.array([1, 1, 0, 0, 0, 1])
    wrench_pf = transform_wrench(wrench_wf, wf_T_tf=wf_T_plane)
    print("wrench_wf: ", wrench_wf)
    print("wrench_pf: ", wrench_pf)
    projected_wrench_wf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf)
    projected_wrench_pf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf, plane_frame_coords=True)
    print("projected_wrench_pf: ", projected_wrench_pf)
    print("projected_wrench_wf: ", projected_wrench_wf)
    projected_wrench_wf_2 = project_wrench_to_plane_2(wrench_wf, w_T_plane, w_T_wf)
    print("projected_wrench_wf_2: ", projected_wrench_wf_2)
    print('CASE 7')
    wrench_wf = np.array([1, 1, 1, 1, 1, 1])
    wrench_pf = transform_wrench(wrench_wf, wf_T_tf=wf_T_plane)
    print("wrench_wf: ", wrench_wf)
    print("wrench_pf: ", wrench_pf)
    projected_wrench_wf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf)
    projected_wrench_pf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf, plane_frame_coords=True)
    print("projected_wrench_pf: ", projected_wrench_pf)
    print("projected_wrench_wf: ", projected_wrench_wf)
    projected_wrench_wf_2 = project_wrench_to_plane_2(wrench_wf, w_T_plane, w_T_wf)
    print("projected_wrench_wf_2: ", projected_wrench_wf_2)

    print('\n\n')

    # TEST ROTATIONS ------------------------
    # CASE 8 - --- Test with rotations
    plane = np.array([0, -1, 0])
    point = np.array([1, 0, 1])
    w_T_plane = get_w_T_plane(plane=plane, point=point)
    w_T_wf = pose_to_matrix(
        np.concatenate([np.array([4, 3, 1]), tr.quaternion_about_axis(0, [0, 0, 1])]))  # 90 degree rotation
    plane_T_wf = np.linalg.inv(w_T_plane) @ w_T_wf
    wf_T_plane = np.linalg.inv(plane_T_wf)
    print("w_T_plane: ", w_T_plane)
    print("plane_T_wf: ", plane_T_wf)
    print('CASE 8')
    wrench_wf = np.array([1, 0, 0, 0, 0, 0])
    wrench_pf = transform_wrench(wrench_wf, wf_T_tf=wf_T_plane)
    print("wrench_wf: ", wrench_wf)
    print("wrench_pf: ", wrench_pf)
    projected_wrench_wf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf)
    projected_wrench_pf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf, plane_frame_coords=True)
    print("projected_wrench_pf: ", projected_wrench_pf)
    print("projected_wrench_wf: ", projected_wrench_wf)
    projected_wrench_wf_2 = project_wrench_to_plane_2(wrench_wf, w_T_plane, w_T_wf)
    print("projected_wrench_wf_2: ", projected_wrench_wf_2)

    print('CASE 9')
    wrench_wf = np.array([0, 1, 0, 0, 0, 0])
    wrench_pf = transform_wrench(wrench_wf, wf_T_tf=wf_T_plane)
    print("wrench_wf: ", wrench_wf)
    print("wrench_pf: ", wrench_pf)
    projected_wrench_wf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf)
    projected_wrench_pf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf, plane_frame_coords=True)
    print("projected_wrench_pf: ", projected_wrench_pf)
    print("projected_wrench_wf: ", projected_wrench_wf)
    projected_wrench_wf_2 = project_wrench_to_plane_2(wrench_wf, w_T_plane, w_T_wf)
    print("projected_wrench_wf_2: ", projected_wrench_wf_2)

    print('CASE 10')
    wrench_wf = np.array([1, 1, 0, 0, 0, 1])
    wrench_pf = transform_wrench(wrench_wf, wf_T_tf=wf_T_plane)
    print("wrench_wf: ", wrench_wf)
    print("wrench_pf: ", wrench_pf)
    projected_wrench_wf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf)
    projected_wrench_pf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf, plane_frame_coords=True)
    print("projected_wrench_pf: ", projected_wrench_pf)
    print("projected_wrench_wf: ", projected_wrench_wf)
    projected_wrench_wf_2 = project_wrench_to_plane_2(wrench_wf, w_T_plane, w_T_wf)
    print("projected_wrench_wf_2: ", projected_wrench_wf_2)

    print('CASE 11')
    wrench_wf = np.array([1, 1, 1, 1, 1, 1])
    wrench_pf = transform_wrench(wrench_wf, wf_T_tf=wf_T_plane)
    print("wrench_wf: ", wrench_wf)
    print("wrench_pf: ", wrench_pf)
    projected_wrench_wf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf)
    projected_wrench_pf = project_wrench_to_plane(wrench_wf, w_T_plane, w_T_wf, plane_frame_coords=True)
    print("projected_wrench_pf: ", projected_wrench_pf)
    print("projected_wrench_wf: ", projected_wrench_wf)
    projected_wrench_wf_2 = project_wrench_to_plane_2(wrench_wf, w_T_plane, w_T_wf)
    print("projected_wrench_wf_2: ", projected_wrench_wf_2)

