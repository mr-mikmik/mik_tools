import numpy as np

from collections import OrderedDict
import gym
import time
from typing import Dict, List, Optional, Sequence, SupportsFloat, Tuple, Type, Union
from gym import logger
from gym.spaces import Discrete
from gym.spaces.space import Space
from gym.utils import seeding


class ConstantSpace(Space):
    """
    A space that always returns the same value.
    Useful as a placeholder when no action is required.
    """

    def __init__(self, value: SupportsFloat):
        self.value = value
        super().__init__(shape=(), dtype=np.float32)

    def sample(self):
        return self.value

    def contains(self, x) -> bool:
        return x == self.value

    def __repr__(self):
        return "ConstantSpace({})".format(self.value)


class ChoiceSpace(Discrete):
    """
    A space that returns one of the given values.
    """

    def __init__(self, choices: List[SupportsFloat]):
        self.choices = choices
        super().__init__(n=len(choices))

    def sample(self, mask=None):
        sample_indxs = super().sample(mask=mask)
        return self.choices[sample_indxs]

    def contains(self, x) -> bool:
        return x in self.choices

    def __repr__(self):
        return "ChoiceSpace({})".format(self.choices)



class EllipsoidSpace(gym.spaces.Space[np.ndarray]):
    def __init__(
            self,
            semiaxis_length: dict,
            dtype: Type = np.float32,
            seed: Optional[Union[int, seeding.RandomNumberGenerator]] = None,
    ):
        self.semiaxis_length = semiaxis_length
        self._keys = list(self.semiaxis_length.keys())
        _A_diag = np.array([self.semiaxis_length[k] for k in self._keys])
        self.A = np.diag(_A_diag)
        self.A_inv = np.diag(np.divide(np.ones_like(_A_diag), _A_diag, out=np.zeros_like(_A_diag), where=_A_diag!=0))
        super().__init__(dtype=dtype, seed=seed)

    def keys(self):
        return self._keys

    def sample(self):
        vec = np.random.randn(len(self.semiaxis_length), 1)
        vec /= np.linalg.norm(vec, axis=0)
        sampled_vec = self.A @ vec
        # add keys
        sample = self._wrap_vector_to_sample(sampled_vec)
        return sample

    def _wrap_vector_to_sample(self, sampled_vec):
        sample = {self._keys[i]: sampled_vec[i].item() for i in range(len(self._keys))}
        return sample

    def contains(self, x) -> bool:
        x_values = np.array([x[k] for k in self._keys])
        vec_sphere = self.A_inv @ x_values
        vec_norm = np.linalg.norm(vec_sphere)
        is_x_contained = np.allclose(vec_norm, np.ones_like(vec_norm))
        return is_x_contained

    def project_action(self, action):
        action_values = np.array([action[k] for k in self._keys])
        vec_equal_axis_space = self.A_inv @ action_values
        vec_norm = vec_equal_axis_space/np.linalg.norm(vec_equal_axis_space)
        vec_proj = self.A @ vec_norm
        sample = self._wrap_vector_to_sample(vec_proj)
        return sample




# DEBUG:
if __name__ == '__main__':
    # semiaxis_length = {'x':0.1, 'y':0.1, 'theta':1.5}
    # ellipsoid_space = EllipsoidSpace(semiaxis_length=semiaxis_length)
    # example_sample = ellipsoid_space.sample()
    # ellipsoid_space.contains(example_sample)

    # TEST choice space
    # values = np.array(['mak', 'mek', 'mik'])
    values = ['mak', 'mek', 'mik']
    choice_space = ChoiceSpace(values)
    example_sample = choice_space.sample()
    print(example_sample)
    print(choice_space.contains('mik'))

