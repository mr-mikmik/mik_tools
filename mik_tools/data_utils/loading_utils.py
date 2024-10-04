import numpy as np
import torch
import os
import pandas as pd
from PIL import Image

import mik_tools.data_utils.data_path_tools as data_path_tools
from mik_tools.camera_tools.pointcloud_utils import load_pointcloud as load_pc


def load_datalegend(data_path):
    dl_path = data_path_tools.get_datalegend_path(data_path)
    if not os.path.isfile(dl_path):
        raise FileNotFoundError(f'Datalenged not found at {dl_path}')
    dl = pd.read_csv(dl_path)
    return dl


def load_array(data_path, scene_name, fc, array_name=None):
    full_path = data_path_tools.get_array_path(save_path=data_path, scene_name=scene_name, fc=fc, array_name=array_name)
    ar = np.load(full_path)
    return ar


def load_tensor(data_path, scene_name, fc, tensor_name=None, device='cpu'):
    full_path = data_path_tools.get_tensor_path(save_path=data_path, scene_name=scene_name, fc=fc, tensor_name=tensor_name)
    tensor = torch.load(full_path, map_location=device, weights_only=True)
    return tensor


def load_point_cloud(data_path, scene_name, camera_name, fc):
    full_path = data_path_tools.get_pointcloud_path(save_path=data_path, scene_name=scene_name, camera_name=camera_name, fc=fc)
    pc_array = load_pc(full_path, as_array=True)
    return pc_array


def load_image_color(data_path, scene_name, camera_name, fc, as_numpy=False):
    full_path = data_path_tools.get_image_color_path(save_path=data_path, scene_name=scene_name,
                                                     camera_name=camera_name, fc=fc, as_numpy=as_numpy)
    file_dir, filename = data_path_tools.split_full_path(full_path)
    filename_only, extension = data_path_tools.split_filename(filename)
    color_array = load_img(file_dir=file_dir, filename=filename_only, extension=extension[1:])
    return color_array


def load_image_depth(data_path, scene_name, camera_name, fc, as_numpy=True):
    full_path = data_path_tools.get_image_depth_path(save_path=data_path, scene_name=scene_name,
                                                     camera_name=camera_name, fc=fc, as_numpy=as_numpy)
    file_dir, filename = data_path_tools.split_full_path(full_path)
    filename_only, extension = data_path_tools.split_filename(filename)
    depth_array = load_img(file_dir=file_dir, filename=filename_only)
    return depth_array


def load_image_depth_filtered(data_path, scene_name, camera_name, fc, as_numpy=True):
    full_path = data_path_tools.get_image_depth_filtered_path(save_path=data_path, scene_name=scene_name,
                                                              camera_name=camera_name, fc=fc, as_numpy=as_numpy)
    file_dir, filename = data_path_tools.split_full_path(full_path)
    filename_only, extension = data_path_tools.split_filename(filename)
    depth_array = load_img(file_dir=file_dir, filename=filename_only)
    return depth_array


def load_shear_deformation(data_path, scene_name, camera_name, fc):
    path = data_path_tools.get_shear_deformation_path(data_path, scene_name=scene_name,
                                                      camera_name=camera_name, fc=fc)
    shear_deformation = np.load(path)
    return shear_deformation


def load_camera_info_depth(data_path, scene_name, camera_name, fc):
    full_path = data_path_tools.get_camera_info_depth_path(save_path=data_path, scene_name=scene_name,
                                                           camera_name=camera_name, fc=fc)
    camera_info = np.load(full_path, allow_pickle=True).item()
    return camera_info


def load_camera_info_color(data_path, scene_name, camera_name, fc):
    full_path = data_path_tools.get_camera_info_color_path(save_path=data_path, scene_name=scene_name,
                                                           camera_name=camera_name, fc=fc)
    camera_info = np.load(full_path, allow_pickle=True).item()
    return camera_info


def load_pressure(data_path, scene_name, camera_name, fc):
    full_path = data_path_tools.get_pressure_path(save_path=data_path, scene_name=scene_name, camera_name=camera_name, fc=fc)
    pressure = np.load(full_path)
    return pressure


def load_wrenches(data_path, scene_name, fc, wrench_name=None):
    full_path = data_path_tools.get_wrench_path(save_path=data_path, scene_name=scene_name, fc=fc, wrench_name=wrench_name)
    wrench_df = pd.read_csv(full_path)
    return wrench_df


def load_tfs(data_path, scene_name, fc, file_name='tfs'):
    full_path = data_path_tools.get_tf_path(save_path=data_path, scene_name=scene_name, fc=fc, file_name=file_name)
    tfs_df = pd.read_csv(full_path)
    return tfs_df


def load_actions(data_path, scene_name, fc):
    full_path = data_path_tools.get_action_path(save_path=data_path, scene_name=scene_name, fc=fc)
    actions_df = pd.read_csv(full_path)
    return actions_df


def load_controller_info(data_path, scene_name, fc):
    full_path = data_path_tools.get_controller_info_path(save_path=data_path, scene_name=scene_name, fc=fc)
    # controller_info is saved as a .npy file, load it
    controller_info = np.load(full_path, allow_pickle=True).item()
    return controller_info


def load_img(file_dir, filename, extension=None):
    # Load an image as a np.ndarray. The image can be saved as .png or .npy
    if extension is None:
        # find what is the file extension (.png, .npy, ...)
        all_files_in_file_dir = [f for f in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, f))]
        our_files_in_file_dir = [f for f in all_files_in_file_dir if filename in f]
        # filter files that start with '.'
        our_files_in_file_dir = [f for f in our_files_in_file_dir if f[0] != '.']
        if len(our_files_in_file_dir) < 1:
            print('File with name {} in dir {} not found'.format(filename, file_dir))
            raise ValueError('File with name {} in dir {} not found'.format(filename, file_dir))
        else:
            our_file = our_files_in_file_dir[0]
    else:
        our_file = '{}.{}'.format(filename, extension)
    path_to_file = os.path.join(file_dir, our_file)
    # read files on dir
    if '.png' in our_file:
        img = Image.open(path_to_file)  # TODO: Test this
        img_array = np.array(img)
    elif '.npy' in our_file:
        img_array = np.load(path_to_file)
    else:
        img_array = None
    return img_array