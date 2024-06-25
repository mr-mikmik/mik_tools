import numpy as np
import pandas as pd

from mik_tools import matrix_to_pose, pose_to_matrix, tr


WRENCH_COLUMNS = ['wrench.force.x', 'wrench.force.y', 'wrench.force.z', 'wrench.torque.x',
                               'wrench.torque.y', 'wrench.torque.z']

TF_COLUMNS = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']


# Wrench data processing ------------------------------------------
def process_raw_wrench_data(wrench_df, frame_id=None, wrench_columns=None):
    if wrench_columns is None:
        wrench_columns = WRENCH_COLUMNS
    frame_ids = wrench_df['header.frame_id'].values
    if frame_id is None:
        # load all the data
        wrench = wrench_df[wrench_columns].values
    elif frame_id in frame_ids:
        # return only the wrench for the given frame id
        wrench = wrench_df[wrench_df['header.frame_id'] == frame_id][wrench_columns].values
    else:
        # frame not found
        print('No frame named {} found. Available frames: {}'.format(frame_id, frame_ids))
        wrench = None
    return wrench


def process_tfs(tfs_df, frame_id=None, ref_id=None):
    raw_tfs = tfs_df
    # add fake tr from parent_frame to parent frame
    parent_frames = raw_tfs['parent_frame'].values
    unique_parent_frames = np.unique(parent_frames)
    id_tr = [0., 0., 0., 0., 0., 0., 1.]  # ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
    parent_frames_data = []
    for parent_frame in unique_parent_frames:
        data_line = [parent_frame, parent_frame] + id_tr
        parent_frames_data.append(data_line)
    parent_frames_df = pd.DataFrame(parent_frames_data, columns=['parent_frame', 'child_frame'] + TF_COLUMNS)
    raw_tfs = pd.concat([raw_tfs, parent_frames_df], ignore_index=True)
    # check the child frames for the frame_id
    tf_column_names = TF_COLUMNS
    frame_ids = raw_tfs['child_frame'].values  # all frame ids
    if frame_id is None:
        # load tf all data
        tfs = raw_tfs[tf_column_names].values
    elif frame_id in frame_ids:
        # return only the tf for the given child frame
        tfs = raw_tfs[raw_tfs['child_frame'] == frame_id][tf_column_names].values
    else:
        # frame not found
        raise NameError('No frame named {} not found. Available frames: {}'.format(frame_id, frame_ids))
    # here tfs are
    if ref_id is not None:
        tr_tfs = []
        # transform the tfs so they are expressed with respect to the frame ref_id
        if ref_id not in frame_ids:
            raise NameError(f'Reference frame named {ref_id} not found found. Available frames: {frame_ids}')
        w_X_ref_id = pose_to_matrix(raw_tfs[raw_tfs['child_frame'] == ref_id][tf_column_names].values[
                                        0])  # TODO: Extenc this for different world reference frames
        for tf_i in tfs:
            w_X_frame_id = pose_to_matrix(tf_i)
            ref_id_X_frame_id = np.linalg.inv(w_X_ref_id) @ w_X_frame_id
            tr_tfs.append(matrix_to_pose(ref_id_X_frame_id))
        tfs = np.stack(tr_tfs, axis=0)
    return tfs
