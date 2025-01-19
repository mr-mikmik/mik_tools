import time

import numpy as np
from mik_tools.env_tools import BaseEnv
from mik_tools.env_tools.spaces import Box, Dict, Discrete
from mik_tools.recording_utils.data_recording_wrappers import DictSelfSavedWrapper


class SimpleEnv(BaseEnv):

    def __init__(self, *args, **kwargs):
        self.state = np.zeros(2)
        super().__init__(*args, **kwargs)
    @classmethod
    def get_name(cls):
        return 'simple_env'

    def reset(self):
        self.state = np.zeros(2)
        super().reset()

    def _get_action_space(self):
        action_space_dict = {
            'action1': Box(low=0, high=1, shape=(1,), dtype=float),
            'action2': Box(low=0, high=1, shape=(1,), dtype=float)
        }
        action_space = Dict(action_space_dict)
        return action_space

    def _get_observation_space(self):
        observation_space_dict = {
            'obs1': Box(low=0, high=1, shape=(2,), dtype=float),
            'obs2': Discrete(n=10)
        }
        observation_space = Dict(observation_space_dict)
        return observation_space

    def _get_observation(self):
        obs = {
            'obs1': self.state,
            'obs2': np.random.randint(0, 10)
        }
        obs = DictSelfSavedWrapper(obs)
        # fake some computation time
        time.sleep(.5)
        return obs

    def _do_action(self, a):
        # update the state
        delta_state = np.concatenate([a['action1'], a['action2']])
        self.state = np.clip(self.state + delta_state, 0, 1)
        info = {}
        return info


# Collect some data:
if __name__ == "__main__":
    from mik_tools.dataset_tools.data_collection import EnvDataCollector
    from mik_tools import get_dataset_path
    dataset_path = get_dataset_path('test_dataset')
    env = SimpleEnv()
    num_samples = 3
    collector = EnvDataCollector(env, data_path=dataset_path, trajectory_length=10, reset_trajectories=True, save_trajectory_on_the_fly=True)
    collector.collect_data(num_samples)