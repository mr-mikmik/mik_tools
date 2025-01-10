import abc
from mik_tools.recording_utils.data_recording_wrappers import ControllerInfoSavedWrapper


class ControllerBase(object):
    def __init__(self, env):
        self.env = env
        self.controller_info = {}

    @classmethod
    def get_name(cls):
        return 'controller_base'

    @property
    def name(self):
        return self.get_name()

    @abc.abstractmethod
    def control(self, obs, info=None):
        pass

    @abc.abstractmethod
    def get_controller_params(self):
        pass

    def is_action_valid(self, action):
        return self.env.is_action_valid(action)

    def reset_env(self):
        out = self.env.reset()
        return out

    def reset_controller_info(self):
        self.controller_info = {}

    def save_controller_info(self, fc, data_save_params):
        if self.controller_info:
            controller_info = ControllerInfoSavedWrapper(self.controller_info, data_params=data_save_params)
            controller_info.modify_data_params(data_save_params)
            controller_info.save_fc(fc=fc)