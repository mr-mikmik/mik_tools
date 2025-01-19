import numpy as np

from collections import OrderedDict
import gym
import time
from typing import Dict, List, Optional, Sequence, SupportsFloat, Tuple, Type, Union
from gym import logger
from gym.spaces import Discrete, Box, Tuple, Dict


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


# DEBUG:
if __name__ == '__main__':
    # TEST choice space
    # values = np.array(['mak', 'mek', 'mik'])
    values = ['mak', 'mek', 'mik']
    choice_space = ChoiceSpace(values)
    example_sample = choice_space.sample()
    print(example_sample)
    print(choice_space.contains('mik'))
