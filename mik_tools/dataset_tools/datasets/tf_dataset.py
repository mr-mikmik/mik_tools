from mik_tools.data_utils.loading_utils import load_image_color, load_tfs
from mik_tools import matrix_to_pose, pose_to_matrix
import numpy as np
import pandas as pd
from ..dataset_base import DatasetBase


class TFDataset(DatasetBase):
    def __init__(self, *args, **kwargs):
        self.tf_column_names = self._get_tf_column_names()
        super().__init__(*args, **kwargs)

    def _get_tf_column_names(self):
        tfs_column_names = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
        return tfs_column_names

    def _load_tfs(self, fc, scene_name):
        # return the saved tfs as DataFrame
        tfs_df = load_tfs(data_path=self.data_path, scene_name=scene_name, fc=fc)
        return tfs_df

    def _get_tfs(self, fc, scene_name, frame_id=None, ref_id=None):
        # frame_id is currently the name of the child frame
        raw_tfs = self._load_tfs(fc, scene_name)
        # add fake tr from parent_frame to parent frame
        parent_frames = raw_tfs['parent_frame'].values
        unique_parent_frames = np.unique(parent_frames)
        id_tr = [0., 0., 0., 0., 0., 0., 1.] # ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
        parent_frames_data = []
        for parent_frame in unique_parent_frames:
            data_line = [parent_frame, parent_frame] + id_tr
            parent_frames_data.append(data_line)
        parent_frames_df = pd.DataFrame(parent_frames_data, columns=['parent_frame', 'child_frame']+self.tf_column_names)
        raw_tfs = pd.concat([raw_tfs, parent_frames_df], ignore_index=True)
        # check the child frames for the frame_id
        tf_column_names = self.tf_column_names
        frame_ids = raw_tfs['child_frame'].values # all frame ids
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
            w_X_ref_id = pose_to_matrix(raw_tfs[raw_tfs['child_frame'] == ref_id][tf_column_names].values[0]) # TODO: Extend this for different world reference frames
            for tf_i in tfs:
                w_X_frame_id = pose_to_matrix(tf_i)
                ref_id_X_frame_id = np.linalg.inv(w_X_ref_id) @ w_X_frame_id
                tr_tfs.append(matrix_to_pose(ref_id_X_frame_id))
            tfs = np.stack(tr_tfs, axis=0)
        return tfs

