import numpy as np

from typing import Dict, List, Optional, Sequence, SupportsFloat, Tuple, Type, Union
from gym.spaces.space import Space


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