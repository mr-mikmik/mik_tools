import torch
import numpy as np
from tqdm import tqdm
import os


class TrasformProcessedDataset(object):
    """
    Overwrites the dataset saved processed data applyting the transformations provided.
    """
    def __init__(self, dataset, trs):
        self.dataset = dataset
        self.trs = trs

    def process(self, init_indx=0, last_indx=0, indxs=None):
        if indxs is None:
            indxs = np.arange(start=init_indx, stop=(last_indx-1)%len(self.dataset)+1)
        for indx in tqdm(indxs):
            sample_i = self.dataset[indx]
            # transform the sample
            for tr in self.trs:
                sample_i = tr(sample_i)
            # overwrite the sample
            save_path_i = os.path.join(self.dataset.processed_data_path, 'data_{}.pt'.format(indx))
            torch.save(sample_i, save_path_i)


def transform_processed_dataset(dataset, trs, init_indx=0, last_indx=0, indxs=None):
    td = TrasformProcessedDataset(dataset, trs)
    td.process(init_indx=init_indx, last_indx=last_indx, indxs=indxs)




