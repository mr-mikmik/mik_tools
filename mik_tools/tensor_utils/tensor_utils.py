import numpy as np
import torch
from typing import Union


def batched_index_select(X:Union[np.ndarray, torch.Tensor], indxs:Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Batch select an array or tensor based on a batched set of indxs
    Args:
        X (np.ndarray or torch.Tensor): of shape (..., K, <extra_dims>)
        indxs (np.ndarray or torch.Tensor): of shape (..., ) ints of between [0, K]
    Returns:
        X_out: of shape (..., <extra_dims>)
    """
    if torch.is_tensor(X):
        X_out = batched_index_select_tensor(X, indxs)
    elif type(X) == np.ndarray:
        X_out = batched_index_select_array(X, indxs)
    else:
        raise TypeError('X must be a torch tensor or a numpy array, but it is {}'.format(type(X)))
    return X_out


# ======================================================================================================================
# ================================== UTILS ================================================================
# ======================================================================================================================


def batched_index_select_array(X:np.ndarray, indxs:np.ndarray) -> np.ndarray:
    """

    Args:
        X (np.ndarray): of shape (..., K, <extra_dims>)
        indxs (np.ndarray): of shape (..., ) ints of between [0, K]
    Returns:
        X_out: of shape (..., <extra_dims>)
    """
    in_shape = X.shape
    batch_dims = indxs.shape
    num_batch_dims = len(batch_dims)
    extra_dims = in_shape[len(batch_dims) + 1:]
    if num_batch_dims == 0:
        X = np.expand_dims(X, axis=0)
        num_batch_dims = 1
    X_r = X.reshape(-1, *in_shape[num_batch_dims:]) # (B, K, <extra_dims>)
    indxs_flat = indxs.flatten() # (B,)
    X_selected = X_r[np.arange(len(indxs_flat)), indxs_flat] # (B, <extra_dims>)
    X_out = X_selected.reshape(*batch_dims, *extra_dims)
    return X_out


def batched_index_select_tensor(X: torch.Tensor, indxs:torch.Tensor) -> torch.Tensor:
    """

    Args:
        X (torch.Tensor): of shape (..., K, <extra_dims>)
        indxs (torch.Tensor): of shape (..., ) ints of between [0, K]
    Returns:
        X_out: of shape (..., <extra_dims>)
    """
    in_shape = X.shape
    batch_dims = indxs.shape
    num_batch_dims = len(batch_dims)
    extra_dims = in_shape[len(batch_dims)+1:]
    if num_batch_dims == 0:
        X = X.unsqueeze(0)
        num_batch_dims = 1
    X_r = X.flatten(end_dim=num_batch_dims-1) # (B, K, <extra_dims>)
    indxs_flat = indxs.flatten() # (B,)
    X_selected = X_r[torch.arange(len(indxs_flat)), indxs_flat] # (B, <extra_dims>)
    X_out = X_selected.reshape(*batch_dims, *extra_dims)
    return X_out


# test
if __name__ == '__main__':
    X = np.arange(24).reshape(2, 3, 4, 1)
    indxs = np.random.randint(0, 4, size=(2, 3))
    Xt = torch.tensor(X)
    indxs_t = torch.tensor(indxs)
    X_selected = batched_index_select_array(X, indxs)
    Xt_selected = batched_index_select_tensor(Xt, indxs_t)
    Xt_selected_2 = torch.gather(Xt, 2, indxs_t.unsqueeze(-1).unsqueeze(-1)).squeeze(-1)
    print('ok')
