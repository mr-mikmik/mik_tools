import numpy as np
import torch
from matplotlib import pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor

from mik_tools.camera_tools.img_utils import project_point_to_image
from mik_tools.camera_tools.sam.sam_utils import get_sam_checkpoint, show_points, show_mask, plot_sam_masks, show_box


class SAMWrapper:

    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.sam = sam_model_registry['vit_h'](checkpoint=get_sam_checkpoint())
        self.sam.to(device)

        self.sam_predictor = SamPredictor(self.sam)

    def segment(self, img: np.ndarray, point_coords: np.ndarray, point_labels: np.ndarray = None, box: np.ndarray = None, vis: bool = False):
        """

        :param img: numpy array of shape (W, H, 3) as format uint8 i.e. 0-255
        :param point_coords: numpy array of shape (K, 2) where K is the number of points
        :param point_labels: numpy of shape (K,) where K is the number of points
        :param vis: <bool> whether to visualize the output
        :return:
            masks: list of numpy arrays of shape (W, H) as format float32 i.e. 0-1
            scores: list of floats where each float is the score of the corresponding mask
            logits: list of numpy arrays of shape (W, H) as format float32 i.e. 0-1
        """
        if len(point_coords.shape) == 1:
            point_coords = point_coords.reshape(-1, 2)
        if point_labels is None:
            point_labels = np.ones(len(point_coords))
        if vis:
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            show_points(point_coords, point_labels, plt.gca())
            if box is not None:
                show_box(box, plt.gca())
            plt.axis('off')
            plt.show()

        self.sam_predictor.set_image(img)
        masks, scores, logits = self.sam_predictor.predict(point_coords=point_coords, point_labels=point_labels, box=box,
                                                           multimask_output=True)

        if vis:
            axes = plot_sam_masks(img, point_coords, point_labels, masks, scores, logits, box=box)
            plt.show()

        return masks, scores, logits

    def segment_3d_point(self, img: np.ndarray, point_xyz: np.ndarray, K: np.ndarray, point_labels: np.ndarray = None, vis: bool = False):
        """
        Project a 3D point to the image and segment the point in the image
        :param img: numpy array of shape (W, H, 3) as format uint8 i.e. 0-255
        :param point_xyz: point in 3D space in the camera frame as numpy array of shape (K,3)
        :param K: camera intrinsic matrix as numpy array of shape (3, 3)
        :param point_labels: numpy of shape (K,) where K is the number of points. If None, it is assumed to be 1
        :param vis: <bool> whether to visualize the output
        :return:

        """
        point_uv = project_point_to_image(point_xyz, K, as_int=True) # (K, 2)
        out = self.segment(img=img, point_coords=point_uv, point_labels=point_labels, vis=vis) # (masks, scores, logits)
        return out

