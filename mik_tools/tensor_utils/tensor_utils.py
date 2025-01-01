import numpy as np
import torch
import torch.nn.functional as F
from typing import Union, Tuple, List


def batched_eye_tensor(n:int, batch_shape:Union[Tuple, List]=(), dtype=None, device=None) -> torch.Tensor:
    """
    Create a batched identity matrix
    Args:
        n (int): size of the square matrix
        batch_shape (Tuple or List): batch shape
    Returns:
        eye (torch.Tensor): of shape (*batch_shape, n, n)
    """
    eye = torch.eye(n, dtype=dtype, device=device).repeat(*batch_shape, 1, 1) # (*batch_shape, n, n)
    return eye


def batched_eye_array(n:int, batch_shape:Union[Tuple, List]=()) -> np.ndarray:
    """
    Create a batched identity matrix
    Args:
        n (int): size of the square matrix
        batch_shape (Tuple or List): batch shape
    Returns:
        eye (np.ndarray): of shape (*batch_shape, n, n)
    """
    B = np.prod(batch_shape)
    eye = np.eye(n, dtype=np.float32) # (n, n)
    # extend the batch dim
    eye = np.expand_dims(eye, axis=0) # (1, n, n)
    eye = np.repeat(eye, B, axis=0) # (B, n, n)
    eye = eye.reshape(*batch_shape, n, n) # (*batch_shape, n, n)
    return eye


def log_barrier(x:torch.Tensor, x_min:torch.Tensor, x_max:torch.Tensor, alpha:float=1.0, epsilon=1e-8) -> torch.Tensor:
    """
    Compute the log barrier function that penalizes the values outside the range (x_min, x_max)
    Note if x==x_min or x==x_max, the function will return inf.
    Args:
        x (torch.Tensor): of shape (..., )
        x_min (torch.Tensor): of shape (..., )
        x_max (torch.Tensor): of shape (..., )
        alpha (float): parameter of the log barrier [0, inf] - the larger, the closer it is to the minimum. About 3.0 is a good value.
    Returns:
        out (torch.Tensor): of shape (..., )
    """
    x_max = x_max + epsilon
    x_min = x_min - epsilon
    x_range = x_max - x_min
    min_scaled = 2 * (x - x_min) / x_range
    max_scaled = 2 * (x_max - x) / x_range
    out = -torch.log(min_scaled) - torch.log(max_scaled)
    out = alpha * out
    # set out to 0 for cases where range is 0
    return out


def soft_maximum_log_barrier(x:torch.Tensor, z:torch.Tensor, alpha:float=1.0) -> torch.Tensor:
    """
    Compute the log barrier function
    Args:
        x (torch.Tensor): of shape (..., )
        alpha (float): parameter of the log barrier [0, inf] - the larger, the closer it is to the minimum. About 3.0 is a good value.
    Returns:
        out (torch.Tensor): of shape (..., )
    """
    # out = z + 1/alpha * torch.log(1 + torch.exp(alpha * (x - z)))
    # This is equivalent as the expression above, but for stability, it resolves to the linear case when (x-z)>threshold
    out = z + F.softplus(x-z, beta=alpha, threshold=10)
    return out


def soft_minimum_log_barrier(x:torch.Tensor, z:torch.Tensor, alpha:float=1.0) -> torch.Tensor:
    """
    Softened minimum function using log-barriers
    :param x: (...,) tensor
    :param z: (...,) tensor
    :param alpha: float [0, inf] - the larger, the closer it is to the minimum. About 3.0 is a good value.
    :return: (...,) tensor
    """
    # y_min_log_barrier = z - 1 / alpha * torch.log(1 + torch.exp(-alpha * (x - z)))
    # This is equivalent as the expression above, but for stability, it resolves to the linear case when (z-x)>threshold
    y_min_log_barrier = z - F.softplus(z-x, beta=alpha, threshold=10)
    return y_min_log_barrier


def soft_relu(x:torch.Tensor, alpha:float=1.0) -> torch.Tensor:
    """
    Softened version of the ReLU function
    :param x: (...,) tensor
    :param alpha: float [0, inf] - the larger, the closer it is to the minimum. About 3.0 is a good value.
    :return: (...,) tensor
    """
    # out = torch.log(1 + torch.exp(alpha*x))/alpha
    out = F.softplus(x, beta=alpha)
    return out


def soft_unit_step(x:torch.Tensor, alpha:float=1.0) -> torch.Tensor:
    """
    Softened version of the unit step function (i.e. Compute the unit step function, i.e. 1 if x>0, 0 otherwise)
    It is like the sigmoid function but with variable control of the slope.
    :param x: (...,) tensor
    :param alpha: float [0, inf] - the larger, the closer it is to the minimum. About 3.0 is a good value.
    :return: (...,) tensor
    """
    # out = 1.0/(1.0 + torch.exp(-alpha*x))
    out = torch.sigmoid(alpha*x) # this is equivalent as the expression above
    return out


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


def compute_bidirectional_mask_intersection_score(mask_1:torch.Tensor, mask_2:torch.Tensor, normalized=True) -> torch.Tensor:
    """
    Compute the intersection (sum) of two masks
    Args:
        mask_1 (torch.Tensor): shape (B, ...)
        mask_2 (torch.Tensor): shape (K, ...)
    Returns:
        iou_scores (torch.Tensor): shape (B, K)
    """
    not_mask_1 = torch.logical_not(mask_1).to(torch.float32)  # (B, M)
    not_mask_2 = torch.logical_not(mask_2).to(torch.float32)  # (B, M)
    intersection_1s = compute_mask_intersection_score(mask_1, mask_2, normalized=normalized)  # (B, K)
    intersection_0s = compute_mask_intersection_score(not_mask_1, not_mask_2, normalized=normalized)  # (B, K)
    intersection = intersection_1s + intersection_0s
    if normalized:
        intersection = 0.5 * intersection # 1 means perfect ovelapping. -- each 1s and 0s count as 50% of the score.
    return intersection


def compute_mask_intersection_score(mask_1:torch.Tensor, mask_2:torch.Tensor, normalized=False) -> torch.Tensor:
    """
    Compute the intersection (sum) of two masks.
        1 would be that there is one mask entirely contained inside the other.
        0 the masks are not overlapping.
    Args:
        mask_1 (torch.Tensor): shape (B, ...)
        mask_2 (torch.Tensor): shape (K, ...)
    Returns:
        iou_scores (torch.Tensor): shape (B, K)
    """
    # TODO: Consider normalize by the maxium overlapping possible.
    B = mask_1.shape[0]
    K = mask_2.shape[0]
    mask_1_flat = mask_1.flatten(start_dim=1)  # (B, M)
    mask_2_flat = mask_2.flatten(start_dim=1)  # (K, M)
    # compute the 1s match (sum how much 1s match)
    intersection = torch.einsum('bm,km->bk', mask_1_flat, mask_2_flat)  # (B, K)
    if normalized:
        mask_1_flat_sum = mask_1_flat.sum(dim=1)  # (B,)
        mask_2_flat_sum = mask_2_flat.sum(dim=1)  # (K,)
        norm_value = torch.min(torch.stack([
            mask_1_flat_sum.unsqueeze(1).repeat_interleave(repeats=K, dim=1),
            mask_2_flat_sum.unsqueeze(0).repeat_interleave(repeats=B, dim=0),
        ], dim=-1), dim=-1)[0]  # (B, K)
        out = torch.nan_to_num(intersection / norm_value)
        # note: nan_to_num avoid the nans if there is the case of division by 0
    else:
        out = intersection
    return out


def compute_mask_iou_score(mask_1:torch.Tensor, mask_2:torch.Tensor, normalized=False, batch_size=None) -> torch.Tensor:
    """
    Compute the iou of two masks.
        1 the masks perfectly intersect to each other, i.e. intersection = union
        0 the masks are not overlapping. -- intersection = 0
    Args:
        mask_1 (torch.Tensor): shape (B, ...)
        mask_2 (torch.Tensor): shape (K, ...)
    Returns:
        iou_scores (torch.Tensor): shape (B, K)
    """
    B = mask_1.shape[0]
    K = mask_2.shape[0]
    mask_1_flat = mask_1.flatten(start_dim=1)  # (B, M)
    mask_2_flat = mask_2.flatten(start_dim=1)  # (K, M)
    # compute the 1s match (sum how much 1s match)
    intersection = torch.einsum('bm,km->bk', mask_1_flat, mask_2_flat)  # (B, K) where each element is the number of intersections
    # union = torch.einsum('bm,km->bkm', mask_1_flat, mask_2_flat).sum(dim=-1)# (B, K) where each element is the number of
    if batch_size is None:
        union = ((mask_1_flat.unsqueeze(1)) | (mask_2_flat.unsqueeze(0))).sum(dim=-1)# (B, K) where each element is the number of
    else:
        union = []
        for i in range(B):
            mask_or_res = ((mask_1_flat[0].unsqueeze(0)) + mask_2_flat) > 0
            union_i = mask_or_res.sum(dim=-1) # (K,)
            union.append(union_i)
        union = torch.stack(union, dim=0) # (B, K)
    iou_scores = torch.nan_to_num(intersection / union)
    return iou_scores


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


def test_batch_index_selection():
    X = np.arange(24).reshape(2, 3, 4, 1)
    indxs = np.random.randint(0, 4, size=(2, 3))
    Xt = torch.tensor(X)
    indxs_t = torch.tensor(indxs)
    X_selected = batched_index_select_array(X, indxs)
    Xt_selected = batched_index_select_tensor(Xt, indxs_t)
    Xt_selected_2 = torch.gather(Xt, 2, indxs_t.unsqueeze(-1).unsqueeze(-1)).squeeze(-1)
    print('ok')


def test_compute_mask_iou_score():
    mask_1 = torch.tensor(np.random.randint(0,2, (2, 3, 3)))
    mask_2 = torch.tensor(np.random.randint(0,2, (1, 3, 3)))
    iou_score = compute_mask_iou_score(mask_1, mask_2)
    print(mask_1)
    print(mask_2)
    print(iou_score)


if __name__ == '__main__':
    # test_batch_index_selection()
    test_compute_mask_iou_score()

