import torch
import os
import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from mik_tools.mik_learning_tools.learning_tools.arg_tools import filter_trainer_args
from mik_tools.mik_learning_tools.learning_tools.model_load_tools import get_checkpoint_path, load_model, \
    filter_model_args


def train_model(args, Model, model_args, data_path, dataset, train_loader, val_loader=None, initialize_model_weights_version=None, resume_training_version=None, callbacks=None):
    data_sizes = dataset.get_sizes()

    if resume_training_version is not None:
        model = load_model(Model, load_version=resume_training_version, data_path=data_path,
                           input_sizes=dataset.get_sizes())
    else:
        model = Model(data_sizes, **model_args)
    # Load model weights
    if initialize_model_weights_version is not None:
        model.load_weights_from_version(initialize_model_weights_version)

    print(f'\n\n\t Training Model {model.name} Started\n\n')
    logger = TensorBoardLogger(os.path.join(data_path, 'tb_logs'), name=model.name)

    # trainer = Trainer.from_argparse_args(args)  # (gpus=gpus, max_epochs=MAX_EPOCHS, logger=logger, log_every_n_steps=1)
    trainer_args = filter_trainer_args(args)
    trainer_args['logger'] = logger
    trainer_args['callbacks'] = callbacks
    trainer = Trainer(**trainer_args)  # (gpus=gpus, max_epochs=MAX_EPOCHS, logger=logger, log_every_n_steps=1)
    if resume_training_version is not None:
        checkpoint_path = get_checkpoint_path(Model.get_name(), load_version=resume_training_version,
                                              data_path=data_path)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=checkpoint_path)
    else:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)