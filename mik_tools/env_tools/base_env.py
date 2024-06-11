from gym import Env
from abc import abstractmethod
import time


class BaseEnv(Env):
    """
    Main Attributes:
     - action_space:
     - observation_space:
     - reward_range: (by default is (-inf, inf))
    Main methods:
     - step(action):
     - reset():
     - render():
     - close():
    For more information about the Env class, check: https://github.com/openai/gym/blob/master/gym/core.py
    """
    def __init__(self):
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()
        self.num_steps = 0
        self._ref_obs = None
        self.ref_obs_timestamp = None # This simplifies checking if the reference observation was updated

    @classmethod
    def get_name(cls):
        return 'base_env'

    @abstractmethod
    def _get_action_space(self):
        pass

    @abstractmethod
    def _get_observation_space(self):
        pass

    @abstractmethod
    def _get_observation(self):
        pass

    def _do_action(self, a):
        return {}

    @property
    def ref_obs(self):
        return self._ref_obs

    @ref_obs.setter
    def ref_obs(self, value):
        self._ref_obs = value
        self.ref_obs_timestamp = time.time()

    def update_ref_obs(self, update_dict):
        if self._ref_obs is not None:
            self._ref_obs.update(update_dict)
        else:
            self._ref_obs = update_dict
        self.ref_obs_timestamp = time.time()

    def update_reference_obs(self):
        self.ref_obs = self._get_ref_obs()
        self.ref_obs_timestamp = time.time()

    def _get_ref_obs(self):
        return None

    def is_action_valid(self, a):
        # by default, all actions are valid.
        # Override this method if some actions are not valid
        return True

    def _is_done(self, observation, a):
        return False

    def get_observation(self):
        obs = self._get_observation()
        return obs

    def get_action(self):
        valid_action = False
        action = None
        for i in range(1000):
            action = self.action_space.sample()
            valid_action = self.is_action_valid(action)
            if valid_action:
                break
        return action, valid_action

    def step(self, a):
        # This is just the basic layout. It can be extended on subclasses
        info = {}
        action_feedback = self._do_action(a)
        info.update(action_feedback)
        observation = self.get_observation()
        done = self._is_done(observation, a)
        reward = self._get_reward(a, observation)
        self.num_steps += 1
        return observation, reward, done, info

    def render(self):
        pass

    def _get_reward(self, a, observation):
        return 0

    def reset(self):
        self.num_steps = 0




