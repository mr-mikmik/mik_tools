import numpy as np
from collections import defaultdict
import torch

from mik_tools.dataset_tools.legended_dataset import LegendedDataset


class TrajectoryDataset(LegendedDataset):
    def __init__(self, *args, scene_name=None, load_trajectories=False, filter_out_done=False, **kwargs):
        self.scene_name = scene_name
        self.filter_out_done = filter_out_done
        self.load_trajectories = load_trajectories
        self.trajectory_lines = None
        self.trajectory_index_key = self._get_trajectory_index_key()
        self.trajectory_step_key = self._get_trajectory_step_key()
        super().__init__(*args, **kwargs)
        if self.load_trajectories:
            self._get_trajectories_lines()

    @property
    def name(self):
        name = super().name
        if self.scene_name is not None:
            # add the scene
            if type(self.scene_name) is str:
                scene_names = self.scene_name
            else:
                scene_names = '_'.join(list(self.scene_name))
            name = name + '_{}'.format(scene_names)
        if self.load_trajectories:
            name = name + '_trajectories'
        return name

    def _slice_dataset(self, item):
        """
        Returns a new dataset where each sample is a sliding window of the original dataset
        :param window_size:
        :param step_size:
        :return:
        """
        # sliced_dataset = super()._slice_dataset(item)
        sliced_dataset = self.copy()  # copy the dataset
        sliced_dataset.item_indxs = self.item_indxs[item]
        sliced_dataset.sample_codes = self.sample_codes[item]
        # also slice the trajectory lines
        if self.trajectory_lines is not None:
            new_trajectory_lines = {k:v for k, v in sliced_dataset.trajectory_lines.items() if k in sliced_dataset.sample_codes}
            sliced_dataset.trajectory_lines = new_trajectory_lines
        return sliced_dataset

    def _subsample_dataset(self, item):
        """
        Returns a new dataset where each sample is a sliding window of the original dataset
        :param item: <list or numpy array> list of indexes to subsample
        :return:
        """
        subsampled_dataset = self.copy()
        # subsample the sample_codes and item_indxs
        subsampled_dataset.item_indxs = [self.item_indxs[i] for i in item]
        subsampled_dataset.sample_codes = [self.sample_codes[i] for i in item]
        if self.trajectory_lines is not None:
            new_trajectory_lines = {k: v for k, v in subsampled_dataset.trajectory_lines.items() if
                                k in subsampled_dataset.sample_codes}
            subsampled_dataset.trajectory_lines = new_trajectory_lines
        return subsampled_dataset

    def _get_sample_codes(self):
        """
        Return a list containing the sample codes.
        Overwrite the function in case the data needs to be filtered.
        By default we load all the sample codes we find in the datalegend. We also filter them by scene if we specify the scene.
        :return:
        """
        dl = self.dl
        if self.load_trajectories:
            # in this case sample codes are trajectory codes instead of line codes
            trajectory_lines = self._get_trajectories_lines(dl=dl)
            sample_codes = list(trajectory_lines.keys())
        else:
            # sample_codes are the lines indexes of the dl_filtered
            sample_codes = dl.index.to_numpy()
        return sample_codes

    def _get_trajectory_index_key(self):
        return 'TrajectoryIndex'

    def _get_trajectory_step_key(self):
        return 'TrajectoryStep'

    def _get_trajectories_lines(self, dl=None):
        trajectory_lines = {}
        if dl is None:
            dl = self.dl
        trajectory_indxs = dl[self.trajectory_index_key]
        trajectory_steps = dl[self.trajectory_step_key]
        unique_trajectories = np.unique(trajectory_indxs)
        # filter out trajectories that are done
        if self.filter_out_done:
            unique_trajectories = self._filter_out_done_trajectories(unique_trajectories, dl=dl)
        for trj_indx in unique_trajectories:
            # NOTE: We load the trajectory in order of the trajectory steps!
            trajectory_lines[trj_indx] = dl[trajectory_indxs == trj_indx].sort_values(self.trajectory_step_key).index.to_list()
        self.trajectory_lines = trajectory_lines
        return trajectory_lines

    def _filter_out_done_trajectories(self, trajectory_indxs, dl=None):
        filtered_trajectory_indxs = []
        if dl is None:
            dl = self.dl
        for trj_indx in trajectory_indxs:
            dl_traj_i = dl[dl[self.trajectory_index_key] == trj_indx]
            # check if any is Done
            if dl_traj_i['Done'].any():
                pass # do not append this trajectory indx
            else:
                filtered_trajectory_indxs.append(trj_indx)
        return filtered_trajectory_indxs

    def _filter_datalegend(self, datalegend):
        # Extend this method if you want to filter the datalegend
        datalegend_filtered = self._filter_dl_by_scene(dl=datalegend)
        return datalegend_filtered

    def _filter_dl_by_scene(self, dl=None, scene_name=None):
        if dl is None:
            dl = self.dl
        dl_filtered = dl.copy()
        if scene_name is None:
            scene_name = self.scene_name
        if scene_name is not None:
            if type(self.scene_name) is str:
                scene_names = [self.scene_name]
            else:
                scene_names = list(self.scene_name)
            dl_filtered = dl_filtered[dl_filtered['Scene'].isin(scene_names)]
        return dl_filtered

    def _pack_trajectory(self, trajectory_samples):
        # trajectory_samples: list of samples for a given trajectory
        # Here we stack the trajectory samples together
        packed_sample = defaultdict(list)
        trajectory_length = len(trajectory_samples)
        for i, sample_i in enumerate(trajectory_samples):
            for k, v in sample_i.items():
                packed_sample[k].append(v)
        # stack them if they are tensors or numpy arrays.
        for k, vs in packed_sample.items():
            v0 = vs[0]
            if type(v0) in [dict]:
                packed_sample[k] = self._pack_trajectory(vs)
            elif type(v0) in [torch.Tensor]:
                packed_sample[k] = torch.stack(vs, dim=0)
            elif type(v0) in [np.ndarray]:
                packed_sample[k] = np.stack(vs, axis=0)
            elif type(v0) in [int, float]:
                packed_sample[k] = torch.tensor(vs)
            else:
                packed_sample[k] = vs

        return packed_sample

    def _process_sample_i(self, indx):
        sample_code_i = self.sample_codes[indx] # this contains the trajectory indxs for the
        if self.load_trajectories:
            # each sample is a collection of samples along the trajectory
            trajectory_sample_codes = self.trajectory_lines[sample_code_i]
            trajectory_samples = []
            for trj_sc in trajectory_sample_codes:
                trj_sample_i = self._get_processed_sample(trj_sc)
                trj_sample_i = self._tr_sample(trj_sample_i)
                trajectory_samples.append(trj_sample_i)
            # stack all the samples that we can
            sample_i = self._pack_trajectory(trajectory_samples)  # TODO: Replace with sample
        else:
            # we load individual samples, i.e. split trajectories if any into steps
            sample_i = self._get_processed_sample(sample_code_i)
            sample_i = self._tr_sample(sample_i)
        self._save_sample(sample_i, indx)