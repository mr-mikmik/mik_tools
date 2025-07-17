from mik_tools.wrapping_utils.wrapping_utils import ClassWrapper
from mik_tools.recording_utils.recording_utils import record_image_color, record_image_depth, record_image, \
    record_camera_info, record_camera_info_color, record_camera_info_depth, record_point_cloud, record_actions, \
    record_shear_deformation, record_shear_image, record_pressure, record_array, record_deformation_image, record_controller_info, record_image_depth_filtered


class DataSelfSavedWrapper(ClassWrapper):
    def __init__(self, data, data_params=None):
        super().__init__(data)
        if data_params is None:
            data_params = {}
        self.data_params = data_params

    @property
    def data(self):
        return self.wrapped_object

    def save_fc(self, fc):
        # Override on implementation
        pass

    def modify_data_params(self, new_data_params):
        self.data_params.update(new_data_params)


class ArraySelfSavedWrapper(DataSelfSavedWrapper):
    def __init__(self, *args, array_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.array_name = array_name

    def save_fc(self, fc):
        save_path = self.data_params['save_path']
        scene_name = self.data_params['scene_name']
        if 'array_name' in self.data_params:
            array_name = self.data_params['array_name']
        else:
            array_name = self.array_name
        record_array(self.data, save_path=save_path, scene_name=scene_name, fc=fc, array_name=array_name)


class PressureSelfSavedWrapper(ArraySelfSavedWrapper):
    def save_fc(self, fc):
        save_path = self.data_params['save_path']
        scene_name = self.data_params['scene_name']
        camera_name = self.data_params['camera_name']
        record_pressure(self.data, save_path=save_path, scene_name=scene_name, camera_name=camera_name, fc=fc)


class ImageSelfSavedWrapper(DataSelfSavedWrapper):

    def save_fc(self, fc):
        save_path = self.data_params['save_path']
        scene_name = self.data_params['scene_name']
        camera_name = self.data_params['camera_name']
        additional_params = {}
        if 'image_name' in self.data_params:
            # If there is a specific image name to save
            additional_params['image_name'] = self.data_params['image_name']
        if 'save_as_numpy' in self.data_params:
            additional_params['save_as_numpy'] = self.data_params['save_as_numpy']
        record_image(self.data, save_path=save_path, scene_name=scene_name, camera_name=camera_name, fc=fc, **additional_params)


class ColorImageSelfSavedWrapper(DataSelfSavedWrapper):

    def save_fc(self, fc):
        save_path = self.data_params['save_path']
        scene_name = self.data_params['scene_name']
        camera_name = self.data_params['camera_name']
        if 'save_as_numpy' in self.data_params:
            record_image_color(self.data, save_path=save_path, scene_name=scene_name, camera_name=camera_name, fc=fc, save_as_numpy=self.data_params['save_as_numpy'])
        else:
            record_image_color(self.data, save_path=save_path, scene_name=scene_name, camera_name=camera_name, fc=fc)


class DepthImageSelfSavedWrapper(DataSelfSavedWrapper):
    def save_fc(self, fc):
        save_path = self.data_params['save_path']
        scene_name = self.data_params['scene_name']
        camera_name = self.data_params['camera_name']
        if 'save_as_numpy' in self.data_params:
            record_image_depth(self.data, save_path=save_path, scene_name=scene_name, camera_name=camera_name, fc=fc, save_as_numpy=self.data_params['save_as_numpy'])
        else:
            record_image_depth(self.data, save_path=save_path, scene_name=scene_name, camera_name=camera_name, fc=fc)


class DepthFilteredImageSelfSavedWrapper(DataSelfSavedWrapper):
    def save_fc(self, fc):
        save_path = self.data_params['save_path']
        scene_name = self.data_params['scene_name']
        camera_name = self.data_params['camera_name']
        if 'save_as_numpy' in self.data_params:
            record_image_depth_filtered(self.data, save_path=save_path, scene_name=scene_name, camera_name=camera_name, fc=fc, save_as_numpy=self.data_params['save_as_numpy'])
        else:
            record_image_depth_filtered(self.data, save_path=save_path, scene_name=scene_name, camera_name=camera_name, fc=fc)


class PointCloudSelfSavedWrapper(DataSelfSavedWrapper):
    def save_fc(self, fc):
        save_path = self.data_params['save_path']
        scene_name = self.data_params['scene_name']
        camera_name = self.data_params['camera_name']
        record_point_cloud(self.data, save_path=save_path, scene_name=scene_name, camera_name=camera_name, fc=fc)


class ShearDeformationSelfSavedWrapper(DataSelfSavedWrapper):
    def save_fc(self, fc):
        save_path = self.data_params['save_path']
        scene_name = self.data_params['scene_name']
        camera_name = self.data_params['camera_name']
        record_shear_deformation(self.data, save_path=save_path, scene_name=scene_name, camera_name=camera_name, fc=fc)


class ShearImageSelfSavedWrapper(DataSelfSavedWrapper):
    def save_fc(self, fc):
        save_path = self.data_params['save_path']
        scene_name = self.data_params['scene_name']
        camera_name = self.data_params['camera_name']
        record_shear_image(self.data, save_path=save_path, scene_name=scene_name, camera_name=camera_name, fc=fc)


class DeformationImageSelfSavedWrapper(DataSelfSavedWrapper):
    def save_fc(self, fc):
        save_path = self.data_params['save_path']
        scene_name = self.data_params['scene_name']
        camera_name = self.data_params['camera_name']
        record_deformation_image(self.data, save_path=save_path, scene_name=scene_name, camera_name=camera_name, fc=fc)


class ActionSelfSavedWrapper(DataSelfSavedWrapper):
    def save_fc(self, fc):
        save_path = self.data_params['save_path']
        scene_name = self.data_params['scene_name']
        record_actions(self.data, save_path=save_path, scene_name=scene_name, fc=fc)


class ControllerInfoSavedWrapper(DataSelfSavedWrapper):
    def save_fc(self, fc):
        save_path = self.data_params['save_path']
        scene_name = self.data_params['scene_name']
        record_controller_info(self.data, save_path=save_path, scene_name=scene_name, fc=fc)


class CameraInfoSelfSavedWrapper(DataSelfSavedWrapper):
    def save_fc(self, fc):
        save_path = self.data_params['save_path']
        scene_name = self.data_params['scene_name']
        camera_name = self.data_params['camera_name']
        info_name = self.data_params.get('info_name', 'camera_info')
        record_camera_info(self.data, save_path=save_path, scene_name=scene_name, camera_name=camera_name, info_name=info_name, fc=fc)


class CameraInfoColorSelfSavedWrapper(DataSelfSavedWrapper):
    def save_fc(self, fc):
        save_path = self.data_params['save_path']
        scene_name = self.data_params['scene_name']
        camera_name = self.data_params['camera_name']
        record_camera_info_color(self.data, save_path=save_path, scene_name=scene_name, camera_name=camera_name, fc=fc)


class CameraInfoDepthSelfSavedWrapper(DataSelfSavedWrapper):
    def save_fc(self, fc):
        save_path = self.data_params['save_path']
        scene_name = self.data_params['scene_name']
        camera_name = self.data_params['camera_name']
        record_camera_info_depth(self.data, save_path=save_path, scene_name=scene_name, camera_name=camera_name, fc=fc)


class DictSelfSavedWrapper(DataSelfSavedWrapper):
    def save_fc(self, fc):
        for data_k, data_v in self.data.items():
            if isinstance(data_v, DataSelfSavedWrapper):
                self.data[data_k].save_fc(fc)

    def modify_data_params(self, new_data_params):
        for data_k, data_v in self.data.items():
            if isinstance(data_v, DataSelfSavedWrapper):
                for k, v in new_data_params.items():
                    self.data[data_k].data_params[k] = v

    def get_self_saved_keys(self):
        self_saved_keys = get_self_saved_keys(self.data)
        return self_saved_keys

    def get_not_self_saved_keys(self):
        not_self_saved_keys = get_not_self_saved_keys(self.data)
        return not_self_saved_keys

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value


def get_self_saved_keys(dict):
    self_saved_keys = []
    for data_k, data_v in dict.items():
        if isinstance(data_v, DataSelfSavedWrapper):
            self_saved_keys.append(data_k)
    return self_saved_keys


def get_not_self_saved_keys(dict):
    not_self_saved_keys = []
    for data_k, data_v in dict.items():
        if not isinstance(data_v, DataSelfSavedWrapper):
            not_self_saved_keys.append(data_k)
    return not_self_saved_keys