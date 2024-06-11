from PIL import Image
import numpy as np
import os
from mik_tools.data_utils.data_path_tools import split_full_path


def make_gif(imgs, save_path, duration=40, loop=0):
    """
    Make a gif from a set of images
    :param imgs: list or collection of images as numpy arrays
    :param save_path: <str> path (must end with .gif)
    :param duration: frame duration in ms
    :param loop: <int> number of loops to loop
    :return:
    """
    # transform to uint8
    if not isinstance(imgs[0], np.ndarray) and imgs[0].dtype != np.uint8:
        imgs = [img.astype(np.uint8) for img in imgs]
    # Transform numpy arrays to PIL images
    ims = [Image.fromarray(img) for img in imgs]
    # Save gif
    save_dir, filename = split_full_path(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ims[0].save(save_path, save_all=True, append_images=ims[1:], duration=duration, loop=loop, optimize=False)
