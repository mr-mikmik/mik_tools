import numpy as np
import torch

from mik_tools.mik_tf_tools.tf_tools import transform_matrix_inverse, transform_points_3d

def project_depth_image(depth_img, K, usvs=None):
    """
    Return xyz coordinates in the optical frame (z-axis is the camera axis)
    Args:
        depth_img: (...,w,h) array or tensor
        K: Intrinsic matrix (...,3,3) array or tensor
    Returns: (..., w, h, 3) array of the (x,y,z) coordiantes for each pixel in the image

    """
    is_tensor = torch.is_tensor(depth_img)
    # reshape to make the operation batched
    input_size = depth_img.shape
    if len(input_size) > 2:
        depth_img = depth_img.reshape(-1, *input_size[-2:])
        K = K.reshape(-1, 3, 3)

    if is_tensor and not torch.is_tensor(K):
        K = torch.from_numpy(K)

    if usvs is None:
        us, vs = get_img_pixel_coordinates(depth_img)
    else:
        us, vs = usvs
        us = us.reshape(depth_img.shape)
        vs = vs.reshape(depth_img.shape)

    xs, ys, zs = project_depth_points(us, vs, depth_img, K)
    if is_tensor:
        # pytorch tensors
        img_xyz = torch.stack([xs, ys, zs], dim=-1)
    else:
        # numpy array
        img_xyz = np.stack([xs, ys, zs], axis=-1)
    img_xyz = img_xyz.reshape(*input_size, 3)
    return img_xyz


def get_img_pixel_coordinates(img):
    # img: tensor or array of size (..., w, h)
    # returns us, vs where us are the h coordinates and vs are the w cooridnates.
    input_shape = img.shape
    if len(input_shape) >= 2:
        img = img.reshape(-1, *input_shape[-2:])
    is_tensor = torch.is_tensor(img)
    w, h = img.shape[-2:]
    batch_size = int(img.shape[0])
    if is_tensor:
        # pytorch tensors
        vs, us = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='ij')
        us = us.unsqueeze(0).repeat_interleave(batch_size, dim=0).reshape(img.shape).to(img.device)
        vs = vs.unsqueeze(0).repeat_interleave(batch_size, dim=0).reshape(img.shape).to(img.device)
    else:
        # numpy arrays
        us, vs = np.meshgrid(np.arange(h), np.arange(w))
        us = np.repeat(np.expand_dims(us, 0), batch_size, axis=0).reshape(img.shape)  # stack as many us as num_batches
        vs = np.repeat(np.expand_dims(vs, 0), batch_size, axis=0).reshape(img.shape)  # stack as many vs as num_batches
    us = us.reshape(*input_shape)
    vs = vs.reshape(*input_shape)
    return us, vs


def project_depth_points(us, vs, depth, K):
    """
    Return xyz coordinates in the optical frame (z-axis is the camera axis)
    Args:
        us: (scalar or any shaped array) image height coordinates (top is 0)
        vs: (scalar or any shaped array, matching us) image width coordinates (left is 0)
        depth: (scalar or any shaped array, matching us) image depth coordinates
        K: Intrinsic matrix (3x3)
    Returns: (scalar or any shaped array) of the (x,y,z) coordinates for each given point

    """
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    try:
        # if input is batched
        w, h = us.shape[-2:]
        num_batch = np.prod(us.shape[:-2])
        is_tensor = torch.is_tensor(K)
        if is_tensor:
            cx = cx.unsqueeze(-1).unsqueeze(-1).repeat_interleave(w, dim=-2).repeat_interleave(h, dim=-1)
            cy = cy.unsqueeze(-1).unsqueeze(-1).repeat_interleave(w, dim=-2).repeat_interleave(h, dim=-1)
            fx = fx.unsqueeze(-1).unsqueeze(-1).repeat_interleave(w, dim=-2).repeat_interleave(h, dim=-1)
            fy = fy.unsqueeze(-1).unsqueeze(-1).repeat_interleave(w, dim=-2).repeat_interleave(h, dim=-1)
        else:
            # numpy case
            cx = np.repeat(np.repeat(np.expand_dims(cx, [-2, -1]), w, axis=-2), h, axis=-1)
            cy = np.repeat(np.repeat(np.expand_dims(cy, [-2, -1]), w, axis=-2), h, axis=-1)
            fx = np.repeat(np.repeat(np.expand_dims(fx, [-2, -1]), w, axis=-2), h, axis=-1)
            fy = np.repeat(np.repeat(np.expand_dims(fy, [-2, -1]), w, axis=-2), h, axis=-1)
    except:
        # code below also works for non-batched inputs
        pass

    xs = (us - cx) * depth / fx
    ys = (vs - cy) * depth / fy
    zs = depth
    return xs, ys, zs


def project_points_pinhole(pc_points, K):
    # TODO: Finish
    pc_xyz = pc_points[..., :3]
    pc_zs = pc_xyz[..., 2:3]
    pc_zs_repeated = np.repeat(pc_zs, 3, axis=-1)
    uvs = ((pc_xyz/pc_zs_repeated) @ K.T )[..., :2]
    uvws = np.concatenate([uvs, pc_zs], axis=-1) # add the depth as w value
    return uvws


def bilinear_interpolate(im, us, vs):
    """
    Bilinearly interpolate from a 2D image with a list of (potentially fractional) image coordinates.
    Adapted from https://stackoverflow.com/a/12729229
    Args:
        im: (w,h) array
        us: (n,) array-like list of height coordinates (top is 0)
        vs: (n,) array-like list of width coordinates (left is 0)
    Returns: (n,) bilinearly interpolated values from im, one for each (u,v) pair
    """
    vs = np.asarray(vs)
    us = np.asarray(us)

    v0 = np.floor(vs).astype(int)
    v1 = v0 + 1
    u0 = np.floor(us).astype(int)
    u1 = u0 + 1

    v0 = np.clip(v0, 0, im.shape[1] - 1)
    v1 = np.clip(v1, 0, im.shape[1] - 1)
    u0 = np.clip(u0, 0, im.shape[0] - 1)
    u1 = np.clip(u1, 0, im.shape[0] - 1)

    Ia = im[v0, u0]
    Ib = im[v1, u0]
    Ic = im[v0, u1]
    Id = im[v1, u1]

    wa = (v1 - vs) * (u1 - us)
    wb = (v1 - vs) * (us - u0)
    wc = (vs - v0) * (u1 - us)
    wd = (vs - v0) * (us - u0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


def project_point_to_image(point_cof, K, as_int=False):
    """
    Return uv image coordinates of the point in the camera frame
    Args:
        point_xyz: (..., 3) array or tensor in the camera frame
        K: Intrinsic matrix (3, 3) array or tensor
    Returns: (..., 2) array of the (u,v) coordiantes for each point

    """
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    point_u = fx * point_cof[..., 0] / point_cof[..., 2] + cx
    point_v = fy * point_cof[..., 1] / point_cof[..., 2] + cy
    if as_int:
        point_u = point_u.round().astype(int)
        point_v = point_v.round().astype(int)
    point_uv = np.stack([point_u, point_v], axis=-1)
    return point_uv


def stereo_vision_triangulation(uv_1, uv_2, w_X_cf1, w_X_cf2, K1, K2):
    """
    Return the 3D coordinates of a point in the world frame given its projections in two cameras
    i.e. stereophotogrammetry
    :param uv_1: (np.ndarray) (2,) coordinates of the point in the first camera frame
    :param uv_2: (np.ndarray) (2,) coordinates of the point in the second camera frame
    :param w_X_cf1: (np.ndarray) (4,4) pose of the first camera in the world frame
    :param w_X_cf2: (np.ndarray) (4,4) pose of the second camera in the world frame
    :param K1: (np.ndarray) (3,3) intrinsic matrix of the first camera
    :param K2: (np.ndarray) (3,3) intrinsic matrix of the second camera
    :return: point_w (np.ndarray) (3,) coordinates of the point in the world frame
    """
    # NOTE: The point uv_1 and the point uv_2 may be not exacly the same point in the world frame,
    # Here we will use the mid-point method to estimate the 3D point
    # First, we will estimate the 3D point in the first camera frame
    point1_cf1 = np.asarray(project_depth_points(uv_1[0], uv_1[1], 1, K1))
    # Second, we will estimate the 3D point in the second camera frame
    point2_cf2 = np.asarray(project_depth_points(uv_2[0], uv_2[1], 1, K2))
    # get the vectors in the world frame
    point1_w = transform_points_3d(point1_cf1, w_X_cf1, points_have_point_dimension=False)
    point2_w = transform_points_3d(point2_cf2, w_X_cf2, points_have_point_dimension=False)
    # compute the two vectors
    p1 = w_X_cf1[:3, 3]
    p2 = w_X_cf2[:3, 3]
    _v1 = point1_w - p1
    _v2 = point2_w - p2
    v1 = _v1 / np.linalg.norm(_v1)
    v2 = _v2 / np.linalg.norm(_v2)
    # get the mid-point
    normal = np.cross(v1, v2)
    n1 = np.cross(v1, normal)
    n2 = np.cross(v2, normal)
    c1 = p1 + np.dot((p2-p1), n2) / np.dot(v1, n2) * v1
    c2 = p2 + np.dot((p1-p2), n1) / np.dot(v2, n1) * v2
    point_w = 0.5 * (c1 + c2)
    return point_w


def mask_mean_uv(mask):
    """
    Obtain the mean uv coordinates of the mask
    :param mask: (np.ndarray) of shape (w,h) of 0 and 1s
    :return: uv_mean (np.ndarray) of shape (2,) the mean uv coordinates of the mask
    """
    mask = mask.astype(np.float32)
    mask /= mask.sum()
    us, vs = get_img_pixel_coordinates(mask)
    uv_mean = np.array([np.sum(us * mask), np.sum(vs * mask)])
    return uv_mean
