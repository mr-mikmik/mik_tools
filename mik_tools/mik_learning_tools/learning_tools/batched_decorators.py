import torch
import torch.nn as nn
import abc


def fake_batched_operation(function, x):
    return x


def batched_1d_operation(function, x):
    """
    Apply a 1d batch operation to an arbitrary shaped tensor (...., num_feats)
    This function handles the tensor reshaping so the function can expect an element of size (N, num_feats)
    :param function:
    :param x:
    :return:
    """
    input_size = x.shape
    if len(x.shape) > 2:
        x = x.reshape(-1, input_size[-1]) # (B, num_feats)
    out = function(x) # (B, ..new_dims) (
    out = out.reshape(*(input_size[:-1] + out.shape[1:]))  # (..., ..new_dims)
    return out


def batched_1d_operation_self(function, x, self):
    """
    Apply a 1d batch operation to an arbitrary shaped tensor (...., num_feats)
    This function handles the tensor reshaping so the function can expect an element of size (N, num_feats)
    :param function:
    :param x:
    :return:
    """
    input_size = x.shape
    if len(x.shape) > 2:
        x = x.reshape(-1, input_size[-1]) # (B, num_feats)
    out = function(self, x) # (B, ..new_dims) (
    out = out.reshape(*(input_size[:-1] + out.shape[1:]))  # (..., ..new_dims)
    return out


def batched_img_operation(function, x):
    """
    Apply a 2d batch operation to an arbitrary shaped tensor (...., num_feats, w, h)
    This function handles the tensor reshaping so the function can expect an element of size (N, num_feats, w, h)
    :param function:
    :param x:
    :return:
    """
    input_size = x.shape
    x = x.reshape(-1, *input_size[-3:]) # (B, num_feats, w, h)
    out = function(x) # (B, ..new_dims..)
    out = out.reshape(*(input_size[:-3] + out.shape[1:])) # (..., ..new_dims)
    return out


def batched_img_operation_self(function, x, self):
    """
    Apply a 2d batch operation to an arbitrary shaped tensor (...., num_feats, w, h)
    This function handles the tensor reshaping so the function can expect an element of size (N, num_feats, w, h)
    :param function:
    :param x:
    :return:
    """
    input_size = x.shape
    x = x.reshape(-1, *input_size[-3:]) # (B, num_feats, w, h)
    out = function(self, x) # (B, ..new_dims..)
    out = out.reshape(*(input_size[:-3] + out.shape[1:])) # (..., ..new_dims)
    return out


def batched_1d_decorator(function):
    """
    Apply a 1d batch operation to an arbitrary shaped tensor (...., num_feats)
    This function handles the tensor reshaping so the function can expect an element of size (N, num_feats)
    """
    def batched_1d_operation_wrapper(x):
        return batched_1d_operation(function, x)

    return batched_1d_operation_wrapper


def batched_1d_method_decorator(function):
    """
    Apply a 1d batch operation to an arbitrary shaped tensor (...., num_feats)
    This function handles the tensor reshaping so the function can expect an element of size (N, num_feats)
    """
    def batched_1d_operation_wrapper(self, x):
        return batched_1d_operation_self(function, x, self=self)

    return batched_1d_operation_wrapper


def batched_img_decorator(function):
    """
    Apply a 2d batch operation to an arbitrary shaped tensor (...., num_feats, w, h)
    This function handles the tensor reshaping so the function can expect an element of size (N, num_feats, w, h)
    """

    def batched_img_operation_wrapper(x):
        return batched_img_operation(function, x)

    return batched_img_operation_wrapper


def batched_img_method_decorator(function):
    """
    Apply a 2d batch operation to an arbitrary shaped tensor (...., num_feats, w, h)
    This function handles the tensor reshaping so the function can expect an element of size (N, num_feats, w, h)
    """

    def batched_img_operation_wrapper(self, x):
        return batched_img_operation(function, x, self=self)

    return batched_img_operation_wrapper


def fake_batched_decorator(function):
    def fake_batched_operation_wrapper(x):
        return fake_batched_operation(function, x)

    return fake_batched_operation_wrapper


@batched_1d_decorator
def sum_1(x):
    print(x.shape)
    out = x + 1
    return out


# TEST:
if __name__ == '__main__':
    x = torch.ones((5, 3, 4))
    print(x.shape)      # (5,3,4)
    out = sum_1(x)      # (15, 4)
    print(out.shape)    # (5,3,4)

