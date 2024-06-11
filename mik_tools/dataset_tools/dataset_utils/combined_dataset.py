import numpy as np
from mik_tools.dataset_tools.dataset_base import DatasetBase


class CombinedDataset(DatasetBase):

    def __init__(self, datasets, data_name='combined_dataset', shuffle=False, only_common_keys=False, only_keys=None, **kwargs):
        self.datasets = self._process_datasets(datasets)
        self.shuffle = shuffle
        self.only_common_keys = only_common_keys
        if self.only_common_keys:
            self.data_keys = self._get_common_keys()
        else:
            self.data_keys = only_keys
        self.data_map = self._get_datamap()
        super().__init__(data_name=data_name, **kwargs)

    def _get_sample_codes(self):
        return np.arange(len(self.data_map), dtype=np.int64)

    def _get_sample(self, sample_code):
        dataset_indx, data_indx = self.data_map[sample_code]
        dataset_i = self.datasets[dataset_indx]
        sample_i = dataset_i[data_indx]
        # Add the dataset information:
        sample_i['dataset'] = dataset_i.name
        sample_i = self._filter_sample(sample_i)
        return sample_i

    def get_name(self):
        return 'combined_dataset'

    def _filter_sample(self, sample):
        # filter out keys that may not be desired
        if self.data_keys is not None:
            # We will only select the samples to filter
            keys_to_remove = [k for k in sample.keys() if k not in self.data_keys]
            for key_to_remove in keys_to_remove:
                sample.pop(key_to_remove)
        return sample

    def _process_datasets(self, datasets):
        if type(datasets) in [tuple, list, np.ndarray]:
            for dataset_i in datasets:
                if not isinstance(dataset_i, DatasetBase):
                    raise AttributeError('Dataset {} -- Is not a Dataset'.format(dataset_i))
            return datasets
        elif isinstance(datasets, DatasetBase):
            return [datasets]
        else:
            raise AttributeError('Datasets provided {} -- Not suported. Only single dataset or a collection of datasets'.format(datasets))
            return None

    def _get_datamap(self):
        datamap = []
        # return a (N,2) array where first column is the dataset index from self.dataset and the second is the data sample index in that dataset
        for d_i , dataset_i in enumerate(self.datasets):
            len_i = len(dataset_i)
            datamap_i = np.stack([d_i*np.ones(len_i), np.arange(len_i)], axis=-1)
            datamap.append(datamap_i)
        datamap = np.concatenate(datamap, axis=0)
        if self.shuffle:
            np.random.shuffle(datamap) # inplace operation
        datamap = datamap.astype(np.int64)
        return datamap

    def _get_common_keys(self):
        # get keys common in all datasets
        all_keys = [d[0].keys() for d in self.datasets]
        common_keys = set(all_keys[0])
        for keys in all_keys[1:]:
            common_keys.intersection_update(keys)
        common_keys = list(common_keys)
        return common_keys


