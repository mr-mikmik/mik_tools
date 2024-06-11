import abc

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from argparse import ArgumentParser
from torch.nn import functional as F

from mik_tools.mik_learning_tools.learning_tools.normalizers import ImageNormalizer, FakeNormalizer, Normalizer1d
from mik_tools.mik_learning_tools.models.aux import FCModule, FakeModel


class AutoEncoderModelBase(pl.LightningModule):
    def __init__(self, item_size, embedding_size=50, encoder_type='linear',
                 decoder_type='linear', activation='relu', lr=1e-4, dataset_params=None,
                 loss_type='mse', no_batch_norm=False, encoder_autoloss_weight=0.1, decoder_autoloss_weight=0.1, **kwargs):
        super().__init__()
        self.item_size = self._get_item_size(item_size)
        self.embedding_size = embedding_size
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.activation = activation
        self.lr = lr
        self.encoder_autoloss_weight = encoder_autoloss_weight
        self.decoder_autoloss_weight = decoder_autoloss_weight
        self.loss_type = loss_type
        self.dataset_params = dataset_params
        self.mse_loss = nn.MSELoss()
        self.item_loss = self._get_item_loss()
        self.encoding_sizes = self._get_encoding_sizes()
        if no_batch_norm:
            self.batch_norm = FakeNormalizer()
        else:
            self.batch_norm = self._get_batch_norm()
        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()

        self.save_hyperparameters()

    @classmethod
    def get_name(cls):
        return 'autoencoder_model_base'

    @property
    def name(self):
        return self.get_name()

    def forward(self, x, **kwargs):
        # This works in normalized domain
        x_embedding = self.encoder(x, **kwargs)
        x_reconstructed = self.decoder(x_embedding, **kwargs)
        return x_reconstructed, x_embedding

    def encode(self, x, normalize=True, **kwargs):
        if normalize:
            x = self._norm_item(x)
        x_embedding = self.encoder(x, **kwargs)
        return x_embedding

    def decode(self, x_embedding, normalize=True, **kwargs):
        x_rec = self.decoder(x_embedding, **kwargs)
        if normalize:
            x_rec = self._denorm_item(x_rec)
        return x_rec

    def batch_norm_item(self, x):
        # useful when we have more dimensions that just the batch size (i.e. trajectories,...)
        norm_input = self.batch_norm(x)
        return norm_input

    def _get_item_size(self, item_size):
        if type(item_size) in [np.int32, int, np.int64]:
            item_size = np.array([item_size])
        elif type(item_size) in [np.ndarray]:
            pass
        else:
            NotImplementedError('item_size provided does not fulfill any of the available types')
        return item_size

    def _norm_item(self, item):
        item_norm = self.batch_norm.normalize(item)
        return item_norm

    def _denorm_item(self, item_norm):
        item = self.batch_norm.denormalize(item_norm)
        return item

    def _get_encoding_sizes(self):
        sizes = {
            'encoder_input': self.item_size,
            'encoder_output': self.embedding_size,
            'decoder_input': self.embedding_size,
            'decoder_output': self.item_size,
        }
        return sizes

    def _get_sample_from_embedding(self, item_embedding):
        return item_embedding

    @abc.abstractmethod
    def _get_batch_norm(self):
        pass

    @abc.abstractmethod
    def _get_item_loss(self):
        pass

    @abc.abstractmethod
    def _get_encoder(self):
        pass

    @abc.abstractmethod
    def _get_decoder(self):
        pass

    @abc.abstractmethod
    def _step(self, batch, batch_idx, phase='train'):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        loss = self._step(train_batch, batch_idx, phase='train')
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self._step(val_batch, batch_idx, phase='val')
        return loss

    def _compute_loss(self, **kwargs):
        reconstruction_loss = self.item_loss.forward_from_kwargs(**kwargs)
        encoder_autoloss = torch.tensor(0.)
        decoder_autoloss = torch.tensor(0.)
        if hasattr(self.encoder.__class__, 'auto_loss') and callable(getattr(self.encoder.__class__, 'auto_loss')):
            encoder_autoloss = self.encoder.auto_loss()
        if hasattr(self.decoder.__class__, 'auto_loss') and callable(getattr(self.decoder.__class__, 'auto_loss')):
            decoder_autoloss = self.decoder.auto_loss()
        loss = reconstruction_loss + self.encoder_autoloss_weight * encoder_autoloss + self.decoder_autoloss_weight * decoder_autoloss
        logs = {'loss': loss, 'reconstruction_loss': reconstruction_loss, 'encoder_autoloss': encoder_autoloss,
                'decoder_autoloss': decoder_autoloss}
        return loss, logs

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--embedding_size", type=int, default=50)
        parser.add_argument("--encoder_type", type=str, default='linear')
        parser.add_argument("--decoder_type", type=str, default='linear')
        parser.add_argument("--loss_type", type=str, default='mse',
                            help='Loss to be applied to compare imprints. Options: ["mse", "ellipsoid", "imprint_chamfer"] Default: "mse"')
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--activation", type=str, default='relu')
        parser.add_argument("--no_batch_norm", action='store_true')
        parser.add_argument("--encoder_autoloss_weight", type=float, default=0.1)
        parser.add_argument("--decoder_autoloss_weight", type=float, default=0.1)
        return parser


class ItemMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, item_pred, item_gth):
        mse_loss = F.mse_loss(item_pred, item_gth)
        return mse_loss

    def forward_from_kwargs(self, **kwargs):
        item_pred = kwargs['item_pred']
        item_gth = kwargs['item_gth']
        return self.forward(item_pred, item_gth)

    @classmethod
    def get_name(cls):
        return 'item_mse_loss'


class VectorAutoEncoderModel(AutoEncoderModelBase):
    def __init__(self, item_size, item_key, **kwargs):
        self.reconstruct_key = item_key
        super().__init__(item_size, **kwargs)
        self.save_hyperparameters()

    @classmethod
    def get_name(cls):
        return 'vector_autoencoder_model'

    def _get_batch_norm(self):
        batch_norm = Normalizer1d(self.item_size[-1])
        return batch_norm

    def _get_item_loss(self):
        loss_types = ['mse']
        if self.loss_type not in loss_types:
            raise NotImplementedError(
                f'Imprint loss type {self.loss_type} not implemented yet. Available: {loss_types}')
        else:
            item_loss = ItemMSELoss()
        return item_loss

    def _get_encoder(self):
        implemented_encoders = ['linear', 'fake']
        if self.encoder_type == 'linear':
            sizes = [self.encoding_sizes['encoder_input'][-1]] + [500] * 2 + [self.encoding_sizes['encoder_output']]
            encoder = FCModule(sizes=sizes, activation=self.activation)
        elif self.encoder_type == 'fake':
            encoder = FakeModel()
        else:
            raise NotImplementedError(f'Encoder type {self.encoder_type} not implemented. Available encoders: {implemented_encoders}')
        return encoder

    def _get_decoder(self):
        implemented_decoders = ['linear', 'fake']
        if self.decoder_type == 'linear':
            sizes = [self.encoding_sizes['decoder_input']] + [500]*2 + [self.encoding_sizes['decoder_output'][-1]]
            decoder = FCModule(sizes=sizes, activation=self.activation)
        elif self.decoder_type == 'fake':
            decoder = FakeModel()
        else:
            raise NotImplementedError(f'Encoder type {self.encoder_type} not implemented. Available encoders: {implemented_decoders}')
        return decoder

    def _step(self, batch, batch_idx, phase='train'):
        item = batch[self.reconstruct_key]

        item_norm = self.batch_norm(item)  # Normalize the input for both the model and loss
        item_reconstructed_norm, item_embedding = self.forward(item_norm)
        item_reconstructed = self._denorm_item(item_reconstructed_norm)
        loss_data_dict = {'item_pred': item_reconstructed,
                          'item_gth': item,
                          'item_embedding': item_embedding}
        loss_data_dict.update({k: v for k, v in batch.items() if k not in loss_data_dict}) #add batch information
        loss, logs = self._compute_loss(**loss_data_dict) # loss computed in the norm space

        self.log('{}_batch'.format(phase), float(batch_idx), on_step=False, on_epoch=True)
        self.log_dict({f"{phase}_{k}": v for k, v in logs.items()}, on_step=False, on_epoch=True)
        return loss

