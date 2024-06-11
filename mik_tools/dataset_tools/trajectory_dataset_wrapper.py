import numpy as np
import torch

from mik_tools.dataset_tools.dataset_base import DatasetBase


class TrajectoryDatasetWrapper(DatasetBase):
    """
    Given a trajectory dataset with samples composed by x_init and x_final, aggregate them as follows
    original_sample:
        - x_init (T, ...)
        - x_final (T, ...)
        - action (T, ...)
    new_sample:
        - x (T+1, ...)
        - action (T, ...)
    """
    def __init__(self, dataset, **kwargs):
        self.wrapped_dataset = dataset
        self.keys_to_aggregate = self._get_keys_to_aggregate()
        self.all_keys_to_aggregate = [f'{key}_init' for key in self.keys_to_aggregate] + [f'{key}_final' for key in self.keys_to_aggregate]
        kwargs['dtype'] = self.wrapped_dataset.dtype
        super().__init__(data_name=self.wrapped_dataset.data_path, **kwargs)

    @classmethod
    def get_name(self):
        return 'trajectory_dataset_wrapper'

    @property
    def name(self):
        name = '{}_trajectory'.format(self.wrapped_dataset.name)
        return name

    def _get_sample_codes(self):
        return np.arange(len(self.wrapped_dataset))

    def _get_sample(self, sample_code):
        sample = {}
        sample_ref = self.wrapped_dataset[sample_code]
        # aggregate the states
        for k, v in sample_ref.items():
            if k not in self.all_keys_to_aggregate:
                sample[k] = v
        # aggregate the keys:
        for key in self.keys_to_aggregate:
            init_value = sample_ref[f'{key}_init']
            if torch.is_tensor(init_value):
                sample[key] = torch.cat([sample_ref[f'{key}_init'], sample_ref[f'{key}_final'][-1:]], dim=0)
            # check if is numpy array
            elif isinstance(init_value, np.ndarray):
                sample[key] = np.concatenate([sample_ref[f'{key}_init'], sample_ref[f'{key}_final'][-1:]], axis=0)
            else:
                raise NotImplementedError(f'Unknown type for {key}')
        return sample

    def _get_keys_to_aggregate(self):
        all_keys = self.wrapped_dataset.get_sizes().keys()
        init_keys = []
        final_keys = []
        for key in all_keys:
            if key.endswith('_init'):
                init_keys.append(key[:-5])
            elif key.endswith('_final'):
                final_keys.append(key[:-6])
            else:
                pass
        keys_to_aggregate = list(set(init_keys).intersection(set(final_keys)))
        return keys_to_aggregate



