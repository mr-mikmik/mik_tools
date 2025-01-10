from .controller_base import ControllerBase


class RandomController(ControllerBase):
    @classmethod
    def get_name(cls):
        return 'random_controller'

    def control(self, obs, info=None):
        action, valid = self.env.get_action()
        return action

    def get_controller_params(self):
        controller_params = {
            'action_space': str(self.env.action_space)
        }
        return controller_params