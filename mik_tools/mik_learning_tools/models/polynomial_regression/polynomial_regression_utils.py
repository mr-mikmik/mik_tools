import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


def polynomial_basis_function(xs: Tensor, d:int) -> Tensor:
    """
    Extends the input array to a series of polynomial basis functions of it.
    Args:
        xs: torch.Tensor (..., num_feats)
        d: Integer representing the degree of the polynomial basis functions
    Returns:
        Xs: torch.Tensor of shape (..., d*num_feats+1) containing the basis functions for the
        i.e. [1, x, x**2, x**3,...., x**d]
    """
    basis = [torch.ones(xs.shape[:-1], dtype=xs.dtype, device=xs.device).unsqueeze(-1)]+[xs ** i for i in range(1, d + 1)]
    Xs = torch.cat(basis, dim=-1) # (..., d*num_feats + 1)
    return Xs