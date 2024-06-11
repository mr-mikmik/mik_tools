import numpy as np
from torch.utils.data import Dataset
from mik_tools.wrapping_utils.wrapping_utils import AttributeWrapper


class LoadedDatasetWrapper(AttributeWrapper):
    """
    This wrapper is for cases where we want to load the entire dataset into memory.
    This is useful for cases where we do not want to load the data samples every time we want to access them.
    """
    def __init__(self, dataset, shuffle=False):
        self.dataset = dataset
        self.shuffle = shuffle
        super().__init__(dataset)
        self.indxs = self._get_indxs()
        self.data = [self.dataset[indx] for indx in self.indxs]  # load all dataset data into memory

    def __len__(self):
        dataset_len = len(self.data)
        return dataset_len

    def __getitem__(self, item):
        if isinstance(item, slice):
            # item is a slice object containing (start, stop, step)
            sliced_dataset = LoadedDatasetWrapper(self.dataset)
            sliced_dataset.data = self.data[item]
            return sliced_dataset
        else:
            return self.data[item]

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def _get_indxs(self):
        indxs = np.arange(len(self.dataset))
        if self.shuffle:
            indxs = np.random.shuffle(indxs)
        return indxs