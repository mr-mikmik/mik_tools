import os
import torch
import argparse


def load_model(Model, load_version, data_path, load_epoch=None, load_step=None, model_name=None, **kwargs):
    # kwargs can be used to override laoded model parameters
    if model_name is None:
        model_name = Model.get_name()
    checkpoint_path = get_checkpoint_path(model_name, load_version, data_path, load_epoch=load_epoch, load_step=load_step)
    model = Model.load_from_checkpoint(checkpoint_path, **kwargs)
    log_path = os.path.join(data_path, 'tb_logs', '{}'.format(model_name), 'version_{}'.format(load_version))
    model.log_dir = log_path # add the log path to the model so it can be used to load saved stuff.
    return model


def get_checkpoint_path(model_name, load_version, data_path, load_epoch=None, load_step=None):
    if model_name is None:
        version_chkp_path = os.path.join(data_path, 'version_{}'.format(load_version), 'checkpoints')
    else:
        version_chkp_path = os.path.join(data_path, 'tb_logs', '{}'.format(model_name),
                                         'version_{}'.format(load_version), 'checkpoints')
    # 
    if load_epoch is None or load_step is None:
        checkpoints_fs = [f for f in os.listdir(version_chkp_path) if
                          os.path.isfile(os.path.join(version_chkp_path, f))]
        checkpoint_path = os.path.join(version_chkp_path, checkpoints_fs[0])
    else:
        checkpoint_path = os.path.join(version_chkp_path, 
                                       'epoch={}-step={}.ckpt'.format(load_epoch, load_step))

    return checkpoint_path


def get_checkpoint(model_name, load_version, data_path, load_epoch=None, load_step=None, device=None):
    checkpoint_path = get_checkpoint_path(model_name, load_version, data_path, load_epoch=load_epoch, load_step=load_step)
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def filter_model_args(args, Model):
    model_parser = Model.add_model_specific_args(argparse.ArgumentParser())
    model_only_parsed_args, _ = model_parser.parse_known_args()
    model_only_parsed_args = vars(model_only_parsed_args)
    model_args = {k: v for k, v in vars(args).items() if k in model_only_parsed_args}
    return model_args

