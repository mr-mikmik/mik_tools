import numpy as np
import torch
import torch.nn as nn
import abc
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from argparse import ArgumentParser

from mik_tools.mik_learning_tools.learning_tools.model_load_tools import load_model, get_checkpoint


class LightningBaseModel(LightningModule, abc.ABC):

    def __init__(self, input_sizes, lr=1e-4, activation='relu', num_imgs_to_log=None, num_epochs_log_imgs=50, dataset_params=None, weight_decay=0., **kwargs):
        super().__init__()
        self.input_sizes = input_sizes
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_imgs_to_log = num_imgs_to_log
        self.num_epochs_log_imgs = num_epochs_log_imgs
        self.dataset_params = dataset_params
        self._log_dir = None
        self.activation = activation
        self.loaded_version = None
        self.save_hyperparameters(ignore=self._get_ignored_hyperparameters())

    @classmethod
    def get_name(cls):
        return 'lightning_base_model'

    @property
    def name(self):
        return self.get_name()

    @property
    def log_dir(self):
        if self._log_dir is not None:
            return self._log_dir
        else:
            return self.logger.log_dir

    @log_dir.setter
    def log_dir(self, value):
        self._log_dir = value

    def load_weights_from_version(self, load_version):
        if load_version is not None:
            checkpoint = get_checkpoint(self.get_name(), load_version, data_path=self.dataset_params['data_name'])
            state_dict = checkpoint['state_dict']
            self.load_state_dict(state_dict, strict=False)
            self.loaded_version = load_version

    def training_step(self, train_batch, batch_idx):
        loss = self._step(train_batch, batch_idx, phase='train')
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self._step(val_batch, batch_idx, phase='val')
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    @abc.abstractmethod
    def _step(self, batch, batch_idx, phase='train'):
        pass

    def _get_ignored_hyperparameters(self):
        """
        Returns: list of 'str' containing the names of the argument hyperparameters to be excluded from logging
        """
        return []

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--weight_decay", type=float, default=0.)
        parser.add_argument("--activation", type=str, default='relu')
        parser.add_argument("--num_imgs_to_log", type=int, default=None)
        parser.add_argument("--num_epochs_log_imgs", type=int, default=50)
        return parser

    @staticmethod
    def _combine_parent_parsers(parent_parser, *parent_classes):
        parsers = []
        parsers.append(parent_classes[0].add_model_specific_args(parent_parser))
        for parent_class in parent_classes[1:]:
            parsers.append(parent_class.add_model_specific_args(ArgumentParser(add_help=False)))
        parser = ArgumentParser(parents=parsers, add_help=False, conflict_handler='resolve')
        return parser


