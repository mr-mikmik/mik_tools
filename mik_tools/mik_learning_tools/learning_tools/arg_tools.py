import inspect
from argparse import ArgumentParser

from mik_tools.mik_learning_tools.datasets.lightning_dataset_wrapper import LightningDatasetWrapper


def get_lightning_datamodule(dataset, **kwargs):
    # filter out arguments not in LightningDatasetWrapper
    current_params = inspect.signature(LightningDatasetWrapper.__init__).parameters
    filtered_kwargs = {k:v for k, v in kwargs.items() if k in current_params}
    lightning_dataset = LightningDatasetWrapper(dataset, **filtered_kwargs)
    return lightning_dataset


def get_datamodule_specific_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("--batch_size", type=int, default=None, help='Number of samples to be processed per iteration. If None, we will use the full dataset size. Default: None')
    parser.add_argument("--seed", type=int, default=0, help='Random seed for the data split. Default: 0')
    parser.add_argument("--num_workers", type=int, default=8, help='Number of processes used to load the data. Default: 8')
    parser.add_argument("--train_fraction", type=float, default=.8, help='Percent of the dataset used for training. Default: 0.8 (80%)')
    parser.add_argument("--val_fraction", type=float, default=None, help='Percent of the dataset used for validation. If None, it uses the remaining data from training and testing. Default: None')
    parser.add_argument("--train_size", type=int, default=None, help='Number of data samples in dataset used for training. If None, we use the train_fraction. Default: None')
    parser.add_argument("--val_size", type=int, default=None, help='Number of data samples in dataset used for training. If None, we use the train_fraction. Default: None')
    parser.add_argument("--test_size", type=int, default=0, help='Number of data samples in dataset used for testing. Default: 0')
    parser.add_argument("--drop_last", type=bool, default=True, help='Option to exclude the last batch that does not have full batch size. Default: True')
    parser.add_argument("--pin_memory", action='store_true', help='If true, the dataloaders automatically puts the fetched data Tensors in pinnend memory -- faster data transfers to CUDA-enabled GPus.. Default: False')
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
