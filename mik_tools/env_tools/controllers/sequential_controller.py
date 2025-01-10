from .controller_base import ControllerBase


class SequentialController(ControllerBase):
    def __init__(self, *args, actions, **kwargs):
        self.actions = actions
        self.action_iter = iter(self.actions)
        super().__init__(*args, **kwargs)

    @classmethod
    def get_name(cls):
        return 'sequential_controller'

    def control(self, obs, info=None):
        try:
            action = next(self.action_iter)
        except StopIteration:
            self.action_iter = iter(self.actions) # Restart the iteration
            action = next(self.action_iter)
        return action

    def get_controller_params(self):
        controller_params = {
            'action_space': self.actions
        }
        return controller_params
