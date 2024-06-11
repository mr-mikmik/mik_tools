import os
import numpy as np
from PIL import Image as imm
import pandas as pd
from collections import defaultdict

from mik_tools.camera_tools.pointcloud_utils import save_pointcloud
import mik_tools.data_utils.data_path_tools as data_path_tools


def record_array(ar, save_path,  scene_name, fc, array_name=None):
    full_path = data_path_tools.get_array_path(save_path=save_path, scene_name=scene_name, fc=fc, array_name=array_name)
    save_path, filename = data_path_tools.split_full_path(full_path)
    filename_only, extension = data_path_tools.split_filename(filename)
    save_array(ar, filename=filename_only, save_path=save_path)


def record_point_cloud(pc, save_path, scene_name, camera_name, fc):
    full_path = data_path_tools.get_pointcloud_path(save_path=save_path, scene_name=scene_name, camera_name=camera_name, fc=fc)
    save_path, filename = data_path_tools.split_full_path(full_path)
    filename_only, extension = data_path_tools.split_filename(filename)
    save_pointcloud(pc, filename=filename_only, save_path=save_path)


def record_image_color(img, save_path, scene_name, camera_name, fc, save_as_numpy=False):
    full_path = data_path_tools.get_image_color_path(save_path=save_path, scene_name=scene_name, camera_name=camera_name, fc=fc, as_numpy=save_as_numpy)
    save_path, filename = data_path_tools.split_full_path(full_path)
    filename_only, extension = data_path_tools.split_filename(filename)
    file_save_path = save_image(img, filename=filename_only, save_path=save_path, save_as_numpy=save_as_numpy)
    return file_save_path


def record_image_depth(img, save_path, scene_name, camera_name, fc, save_as_numpy=False):
    full_path = data_path_tools.get_image_depth_path(save_path=save_path, scene_name=scene_name,
                                                     camera_name=camera_name, fc=fc, as_numpy=save_as_numpy)
    save_path, filename = data_path_tools.split_full_path(full_path)
    filename_only, extension = data_path_tools.split_filename(filename)
    save_image(img, filename=filename_only, save_path=save_path, save_as_numpy=save_as_numpy)


def record_image_depth_filtered(img, save_path, scene_name, camera_name, fc, save_as_numpy=False):
    full_path = data_path_tools.get_image_depth_filtered_path(save_path=save_path, scene_name=scene_name,
                                                     camera_name=camera_name, fc=fc, as_numpy=save_as_numpy)
    save_path, filename = data_path_tools.split_full_path(full_path)
    filename_only, extension = data_path_tools.split_filename(filename)
    save_image(img, filename=filename_only, save_path=save_path, save_as_numpy=save_as_numpy)


def record_shear_deformation(shear_deformation, save_path, scene_name, camera_name, fc):
    full_path = data_path_tools.get_shear_deformation_path(save_path=save_path, scene_name=scene_name, camera_name=camera_name, fc=fc)
    save_path, filename = data_path_tools.split_full_path(full_path)
    filename_only, extension = data_path_tools.split_filename(filename)
    save_shear_deformation(shear_deformation, filename=filename_only, save_path=save_path)


def record_shear_image(shear_image, save_path, scene_name, camera_name, fc):
    full_path = data_path_tools.get_shear_image_path(save_path=save_path, scene_name=scene_name, camera_name=camera_name, fc=fc)
    save_path, filename = data_path_tools.split_full_path(full_path)
    filename_only, extension = data_path_tools.split_filename(filename)
    save_image(shear_image, filename=filename_only, save_path=save_path, save_as_numpy=False)


def record_pressure(pressure, save_path, scene_name, camera_name, fc):
    full_path = data_path_tools.get_pressure_path(save_path=save_path, scene_name=scene_name, camera_name=camera_name, fc=fc)
    save_path, filename = data_path_tools.split_full_path(full_path)
    filename_only, extension = data_path_tools.split_filename(filename)
    save_array(pressure, filename=filename_only, save_path=save_path)


def record_deformation_image(shear_image, save_path, scene_name, camera_name, fc):
    full_path = data_path_tools.get_deformation_image_path(save_path=save_path, scene_name=scene_name, camera_name=camera_name, fc=fc)
    save_path, filename = data_path_tools.split_full_path(full_path)
    filename_only, extension = data_path_tools.split_filename(filename)
    save_image(shear_image, filename=filename_only, save_path=save_path, save_as_numpy=False)


def record_camera_info_depth(camera_info, save_path, scene_name, camera_name, fc):
    full_path = data_path_tools.get_camera_info_depth_path(save_path=save_path, scene_name=scene_name, camera_name=camera_name, fc=fc)
    save_path, filename = data_path_tools.split_full_path(full_path)
    filename_only, extension = data_path_tools.split_filename(filename)
    save_camera_info(camera_info, filename=filename_only, save_path=save_path)


def record_camera_info_color(camera_info, save_path, scene_name, camera_name, fc):
    full_path = data_path_tools.get_camera_info_color_path(save_path=save_path, scene_name=scene_name,
                                                           camera_name=camera_name, fc=fc)
    save_path, filename = data_path_tools.split_full_path(full_path)
    filename_only, extension = data_path_tools.split_filename(filename)

    save_camera_info(camera_info, filename=filename_only, save_path=save_path)


def record_actions(actions, save_path, scene_name, fc):
    actions_df = pd.DataFrame(actions)
    full_path = data_path_tools.get_action_path(save_path=save_path, scene_name=scene_name, fc=fc)
    save_path, filename = data_path_tools.split_full_path(full_path)
    filename_only, extension = data_path_tools.split_filename(filename)
    save_actions(actions_df, filename=filename_only, save_path=save_path)


def record_controller_info(controller_info, save_path, scene_name, fc):
    full_path = data_path_tools.get_controller_info_path(save_path=save_path, scene_name=scene_name, fc=fc)
    save_path, filename = data_path_tools.split_full_path(full_path)
    filename_only, extension = data_path_tools.split_filename(filename)
    save_controller_info(controller_info, filename=filename_only, save_path=save_path)


def save_image(img, filename, save_path, save_as_numpy=False):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if save_as_numpy:
        file_save_path = os.path.join(save_path, '{}.npy'.format(filename))
        np.save(file_save_path, img)
    else:
        file_save_path = os.path.join(save_path, '{}.png'.format(filename))
        mode = {
            ('uint8', 3): 'RGB',
            ('uint8', 1): 'L',
            ('uint16', 1): 'I;16',
            ('int32', 1): 'I',
            ('uint32', 1): 'I',
            ('float32', 1): 'F',  # TODO: Fix the I for float32
            ('uint16', 3): 'BGR;16',
            ('uint32', 3): 'BGR;32',
        }
        # All modes available: https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
        mode_i = (str(img.dtype), img.shape[-1])
        if not mode_i in mode:
            raise NotImplementedError('Mode {} not supported yet. We support: {}'.format(mode_i, mode.keys()))
        try:
            pil_img = imm.fromarray(img, mode=mode[mode_i])
            pil_img.save(file_save_path)
        except (ValueError, TypeError) as e:
            file_save_path = os.path.join(save_path, '{}.npy'.format(filename))
            np.save(file_save_path, img)
    return file_save_path


def save_shear_deformation(shear_deformation, filename, save_path):
    filename = '{}.npy'.format(filename)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_save_path = os.path.join(save_path, filename)
    np.save(file_save_path, shear_deformation)


def save_camera_info(camera_info, filename, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_path = os.path.join(save_path, '{}.npy'.format(filename))
    np.save(file_path, camera_info)


def save_actions(actions_df, filename, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    full_save_path = os.path.join(save_path, '{}.csv'.format(filename))
    actions_df.to_csv(full_save_path, index=False)


def save_wrenches(wrenches_df, filename, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    full_save_path = os.path.join(save_path, '{}.csv'.format(filename))
    wrenches_df.to_csv(full_save_path, index=False)


def save_array(array, filename, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    full_save_path = os.path.join(save_path, '{}.npy'.format(filename))
    np.save(full_save_path, array)


def save_controller_info(controller_info, filename, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    full_save_path = os.path.join(save_path, '{}.npy'.format(filename))
    np.save(full_save_path, controller_info)

