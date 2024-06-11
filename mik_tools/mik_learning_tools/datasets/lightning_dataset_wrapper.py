import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from mik_tools.mik_learning_tools.learning_tools.arg_tools import get_datamodule_specific_args


class LightningDatasetWrapper(pl.LightningDataModule):

    def __init__(self, dataset, batch_size=None, val_batch_size=None, seed=0, train_fraction=0.8, val_fraction=None, train_size=None,
                 val_size=None, test_size=0, drop_last=True, num_workers=8, pin_memory=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        if self.val_batch_size is None:
            self.val_batch_size = self.batch_size
        self.seed = seed
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_data = None # Filled on setup
        self.val_data = None   # Filled on setup
        self.test_data = None  # Filled on setup
        super().__init__()

    def setup(self, stage=None):
        # Split the data:
        train_size = self.train_size
        if train_size is None:
            train_size = int(self.train_fraction * len(self.dataset))
        if self.val_size is None:
            test_size = self.test_size
            if self.val_fraction is None:
                val_size = len(self.dataset) - train_size - test_size
            else:
                val_size = int(self.val_fraction * len(self.dataset))
        else:
            val_size = self.val_size
            test_size = len(self.dataset) - train_size - val_size
        self.train_data, self.val_data, self.test_data = random_split(self.dataset, [train_size, val_size, test_size],
                                                 generator=torch.Generator().manual_seed(self.seed))

    def train_dataloader(self):
        train_loader = self._get_loader(self.train_data, self.batch_size)
        return train_loader

    def val_dataloader(self):
        val_loader = self._get_loader(self.val_data, self.batch_size)
        return val_loader

    def test_dataloader(self):
        test_loader = self._get_loader(self.test_data, self.batch_size)
        return test_loader

    def _get_loader(self, data, batch_size):
        if batch_size is None:
            batch_size = len(data)
        batch_size = min(batch_size, len(data))
        loader = DataLoader(data, batch_size=batch_size, num_workers=self.num_workers, drop_last=self.drop_last, pin_memory=self.pin_memory)
        return loader

    @staticmethod
    def add_specific_args(parent_parser):
        parser = get_datamodule_specific_args(parent_parser)
        return parser


