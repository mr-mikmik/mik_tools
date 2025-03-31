import os
import re
import numpy as np


"""
This file defines the paths to where each data is saved or loaded.
General things:
  - save_path refers to the directory location that contains all the data (directory where the datalegend may be found)
"""

include_scene_name_to_file_name = False # setting this to false, helps better transfer data accross datasets since it does not need to modify each individual file name
# include_scene_name_to_file_name = True # setting this to true, helps differentiating to which dataset scene belongs each data point.


def split_full_path(full_path):
    path, filename = os.path.split(full_path)
    return path, filename


def split_filename(filename):
    dot_split = os.path.splitext(filename)
    filename_only = dot_split[0]
    extension = dot_split[1]
    return filename_only, extension


def get_datalegend_path(data_path):
    data_dir, data_name = split_full_path(data_path)
    dl_path = os.path.join(data_path, f'{data_name}_DataLegend.csv')
    return dl_path


def get_array_path(save_path, scene_name, fc, array_name=None):
    if array_name is None:
        array_name = 'array_data'
    save_path = os.path.join(save_path, scene_name, array_name)
    if include_scene_name_to_file_name:
        filename = '{}_array_{:06d}'.format(scene_name, fc)
    else:
        filename = 'array_{:06d}'.format(fc)
    file_name = '{}.npy'.format(filename)
    full_path = os.path.join(save_path, file_name)
    return full_path


def get_tensor_path(save_path, scene_name, fc, tensor_name=None):
    if tensor_name is None:
        tensor_name = 'tensor_data'
    save_path = os.path.join(save_path, scene_name, tensor_name)
    if include_scene_name_to_file_name:
        filename = '{}_tensor_{:06d}'.format(scene_name, fc)
    else:
        filename = 'tensor_{:06d}'.format(fc)
    file_name = '{}.pt'.format(filename)
    full_path = os.path.join(save_path, file_name)
    return full_path


def get_pointcloud_path(save_path, scene_name, camera_name, fc):
    save_path = os.path.join(save_path, scene_name, camera_name, 'point_cloud_data')
    if include_scene_name_to_file_name:
        filename = '{}_pc_{:06d}'.format(scene_name, fc)
    else:
        filename = 'pc_{:06d}'.format(fc)
    file_name = '{}.ply'.format(filename)
    full_path = os.path.join(save_path, file_name)
    return full_path


def get_image_path(save_path, scene_name, camera_name, fc, image_name='image', as_numpy=False):
    if include_scene_name_to_file_name:
        filename = '{}_{}_{:06d}'.format(scene_name, image_name, fc)
    else:
        filename = '{}_{:06d}'.format(image_name, fc)
    save_path = os.path.join(save_path, scene_name, camera_name, '{}_data'.format(image_name))
    if as_numpy:
        full_path = os.path.join(save_path, '{}.npy'.format(filename))
    else:
        full_path = os.path.join(save_path, '{}.png'.format(filename))
    return full_path


def get_image_color_path(save_path, scene_name, camera_name, fc, as_numpy=False):
    if include_scene_name_to_file_name:
        filename = '{}_color_{:06d}'.format(scene_name, fc)
    else:
        filename = 'color_{:06d}'.format(fc)
    save_path = os.path.join(save_path, scene_name, camera_name, 'color_data')
    if as_numpy:
        full_path = os.path.join(save_path, '{}.npy'.format(filename))
    else:
        full_path = os.path.join(save_path, '{}.png'.format(filename))
    return full_path


def get_image_depth_path(save_path, scene_name, camera_name, fc, as_numpy=False):
    if include_scene_name_to_file_name:
        filename = '{}_depth_{:06d}'.format(scene_name, fc)
    else:
        filename = 'depth_{:06d}'.format(fc)
    save_path = os.path.join(save_path, scene_name, camera_name, 'depth_data')
    if as_numpy:
        full_path = os.path.join(save_path, '{}.npy'.format(filename))
    else:
        full_path = os.path.join(save_path, '{}.png'.format(filename))
    return full_path


def get_image_depth_filtered_path(save_path, scene_name, camera_name, fc, as_numpy=False):
    if include_scene_name_to_file_name:
        filename = '{}_depth_{:06d}'.format(scene_name, fc)
    else:
        filename = 'depth_{:06d}'.format(fc)
    save_path = os.path.join(save_path, scene_name, camera_name, 'depth_filtered_data')
    if as_numpy:
        full_path = os.path.join(save_path, '{}.npy'.format(filename))
    else:
        full_path = os.path.join(save_path, '{}.png'.format(filename))
    return full_path


def get_camera_info_color_path(save_path, scene_name, camera_name, fc):
    if fc is None:
        suffix = ''
    elif type(fc) is str:
        suffix = ''
        if fc != '':
            suffix = '_{}'.format(fc)
    else:
        suffix = '_{:06d}'.format(fc)
    save_path = os.path.join(save_path, scene_name, camera_name, 'camera_info')
    if include_scene_name_to_file_name:
        filename = '{}_camera_info_color{}'.format(scene_name, suffix)
    else:
        filename = 'camera_info_color{}'.format(suffix)
    full_path = os.path.join(save_path, '{}.npy'.format(filename))
    return full_path


def get_camera_info_depth_path(save_path, scene_name, camera_name, fc):
    if fc is None:
        suffix = ''
    elif type(fc) is str:
        suffix = ''
        if fc != '':
            suffix = '_{}'.format(fc)
    else:
        suffix = '_{:06d}'.format(fc)
    save_path = os.path.join(save_path, scene_name, camera_name, 'camera_info')
    if include_scene_name_to_file_name:
        filename = '{}_camera_info_depth{}'.format(scene_name, suffix)
    else:
        filename = 'camera_info_depth{}'.format(suffix)
    full_path = os.path.join(save_path, '{}.npy'.format(filename))
    return full_path


def get_shear_deformation_path(save_path, scene_name, camera_name, fc):
    if include_scene_name_to_file_name:
        filename = '{}_shear_{:06d}'.format(scene_name, fc)
    else:
        filename = 'shear_{:06d}'.format(fc)
    save_path = os.path.join(save_path, scene_name, camera_name, 'shear_deformation_data')
    full_path = os.path.join(save_path, '{}.npy'.format(filename))
    return full_path


def get_shear_image_path(save_path, scene_name, camera_name, fc):
    if include_scene_name_to_file_name:
        filename = '{}_shear_img_{:06d}'.format(scene_name, fc)
    else:
        filename = 'shear_img_{:06d}'.format(fc)
    save_path = os.path.join(save_path, scene_name, camera_name, 'shear_image_data')
    full_path = os.path.join(save_path, '{}.png'.format(filename))
    return full_path


def get_deformation_image_path(save_path, scene_name, camera_name, fc):
    if include_scene_name_to_file_name:
        filename = '{}_deformation_img_{:06d}'.format(scene_name, fc)
    else:
        filename = 'deformation_img_{:06d}'.format(fc)
    save_path = os.path.join(save_path, scene_name, camera_name, 'deformation_image_data')
    full_path = os.path.join(save_path, '{}.png'.format(filename))
    return full_path


def get_pressure_path(save_path, scene_name, camera_name, fc):
    if include_scene_name_to_file_name:
        filename = '{}_pressure_{:06d}'.format(scene_name, fc)
    else:
        filename = 'pressure_{:06d}'.format(fc)
    save_path = os.path.join(save_path, scene_name, camera_name, 'pressure_data')
    full_path = os.path.join(save_path, '{}.npy'.format(filename))
    return full_path


def get_wrench_path(save_path, scene_name, fc, wrench_name=None):
    if wrench_name is None:
        wrench_name = 'wrenches'
    if include_scene_name_to_file_name:
        filename = '{}_wrench_{:06d}'.format(scene_name, fc)
    else:
        filename = 'wrench_{:06d}'.format(fc)
    save_path = os.path.join(save_path, scene_name, wrench_name)
    full_path = os.path.join(save_path, '{}.csv'.format(filename))
    return full_path


def get_tf_path(save_path, scene_name, fc, file_name='tfs'):
    if include_scene_name_to_file_name:
        filename = '{}_{}_{:06d}'.format(scene_name, file_name, fc)
    else:
        filename = '{}_{:06d}'.format(file_name, fc)
    save_path = os.path.join(save_path, scene_name, 'tfs')
    full_path = os.path.join(save_path, '{}.csv'.format(filename))
    return full_path


def get_output_trajectory_gif_path(save_path, camera_name, traj_indx, file_name='trajectory', scene_name=None):
    if include_scene_name_to_file_name and scene_name is not None:
        filename = '{}_{}_{:06d}'.format(scene_name, file_name, traj_indx)
    else:
        filename = '{}_{:06d}'.format(file_name, traj_indx)
    if scene_name is not None:
        save_path = os.path.join(save_path, 'output_data', 'gifs', scene_name, camera_name)
    else:
        save_path = os.path.join(save_path, 'output_data', 'gifs', camera_name)
    full_path = os.path.join(save_path, '{}.gif'.format(filename))
    return full_path


def get_action_path(save_path, scene_name, fc):
    if include_scene_name_to_file_name:
        filename = '{}_actions_{:06d}'.format(scene_name, fc)
    else:
        filename = 'actions_{:06d}'.format(fc)
    save_path = os.path.join(save_path, scene_name, 'actions')
    full_path = os.path.join(save_path, '{}.csv'.format(filename))
    return full_path


def get_controller_info_path(save_path, scene_name, fc):
    if include_scene_name_to_file_name:
        filename = '{}_controller_info_{:06d}'.format(scene_name, fc)
    else:
        filename = 'controller_info_{:06d}'.format(fc)
    save_path = os.path.join(save_path, scene_name, 'controller_info')
    full_path = os.path.join(save_path, '{}.npy'.format(filename))
    return full_path


def get_filecodes(path):
    filecodes = []
    if not os.path.exists(path):
        return filecodes
    onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    for file_name in onlyfiles:
        rex = re.search('_\d\d\d\d\d\d.', file_name)
        filecode_i = int(rex.group(0)[1:-1])
        filecodes.append(filecode_i)
    return filecodes


def get_next_filecode(path):
    filecodes = get_filecodes(path)
    if len(filecodes) == 0:
        next_filecode = 0
    else:
        max_filecode = np.max(filecodes)
        next_filecode = max_filecode + 1
    return next_filecode

