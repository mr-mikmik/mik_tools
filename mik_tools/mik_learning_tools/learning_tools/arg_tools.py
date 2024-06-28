import inspect
from argparse import ArgumentParser

from mik_tools.mik_learning_tools.datasets.lightning_dataset_wrapper import LightningDatasetWrapper
from mik_tools.mik_learning_tools.datasets.lightning_dataset_wrapper import get_datamodule_specific_args as _get_datamodule_specific_args


def get_lightning_datamodule(dataset, **kwargs):
    # filter out arguments not in LightningDatasetWrapper
    current_params = inspect.signature(LightningDatasetWrapper.__init__).parameters
    filtered_kwargs = {k:v for k, v in kwargs.items() if k in current_params}
    lightning_dataset = LightningDatasetWrapper(dataset, **filtered_kwargs)
    return lightning_dataset


def get_datamodule_specific_args(parent_parser):
    parser = _get_datamodule_specific_args(parent_parser)
    return parser


def get_trainer_specific_args(parent_parser):
    # This is a temporary fix for lightning 2.0.0 update on Trainer before CLI gets more stable
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("--devices", type=str, default='auto', help='The devices to use. Can be set to a positive number (int or str), a sequence of device indices (list or str), the value -1 to indicate all available devices should be used, or "auto" for automatic selection based on the chosen accelerator. Default: "auto".')
    parser.add_argument("--accelerator", type=str, default='auto', help='Supports passing different accelerator types (“cpu”, “gpu”, “tpu”, “ipu”, “hpu”, “mps”, “auto”) as well as custom accelerator instances.. Default: "auto".')
    parser.add_argument("--max_epochs", type=int, default=-1, help="Stop training once this number of epochs is reached. Disabled by default (None). If both max_epochs and max_steps are not specified, defaults to max_epochs = 1000. To enable infinite training, set max_epochs = -1")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulates gradients over k batches before stepping the optimizer. Default: 1.")
    parser.add_argument("--log_every_n_steps", type=int, default=50, help="How often to log within steps. Default: 50.")
    return parser


def filter_trainer_args(args):
    # return trainer only args as a dictionary
    trainer_parser = get_trainer_specific_args(ArgumentParser())
    trainer_args = {k:v for k,v in vars(args).items() if f'--{k}' in trainer_parser._option_string_actions}
    return trainer_args
