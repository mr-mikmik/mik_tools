import numpy as np
import torch
from typing import Tuple, Union, List


def einsum(einum_keys:str, x1:Union[np.ndarray, torch.Tensor], *args) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(x1, torch.Tensor):
        return torch.einsum(einum_keys, x1, *args)
    elif isinstance(x1, np.ndarray):
        return np.einsum(einum_keys, x1, *args)
    else:
        raise NotImplementedError
