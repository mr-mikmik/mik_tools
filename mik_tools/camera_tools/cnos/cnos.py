import numpy as np
import torch
from typing import Tuple
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except (ModuleNotFoundError, ImportError):
    pass

from mik_tools.camera_tools.sam import SAMWrapper, SAM2Wrapper
from mik_tools.camera_tools.sam.sam_utils import show_anns
from mik_tools.camera_tools.dino import DINOWrapper
from mik_tools import matrix_to_pose, pose_to_matrix, tr, transform_matrix_inverse, eye_pose, get_dataset_path
from mik_tools.camera_tools.dino.dino_utils import dino_process_img
from mik_tools.tensor_utils import compute_mask_iou_score


class CNOS(object):
    """
    Class to compute the similarity between two images using the CNOS metric
    """

    def __init__(self, object_rendered_dataset, dino_model_name='dinov2_vits14', device='cuda', sam_type='sam2', min_mask_size=300, max_mask_size=None):
        self.object_rendered_dataset = object_rendered_dataset
        self.dino_model_name = dino_model_name
        self.device = device
        self.sam_type = sam_type
        self.min_mask_size = min_mask_size
        self.max_mask_size = max_mask_size
        self.sam_wrapper = self._init_sam_wrapper()
        self.sam_generator = self._init_sam_generator()
        self.dino_wrapper = self._init_dino()
        self.renered_zs, self.rendered_imgs = self._embed_rendered_dataset()

    def mask(self, img: np.ndarray, visualize=False, top_k=4) -> np.ndarray:
        """
        Mask an image
        Args:
            img: <np.ndarray> of shape (3, h, w)
        Returns:
            mask: <np.ndarray> of shape (1, h, w)
        """
        sam_out = self.sam_generator.generate(img.transpose(1, 2, 0)) # sam expects input format as (h, w, 3)
        # filter masks
        sam_out = filter_small_masks(sam_out, self.min_mask_size, self.max_mask_size)
        # obtain the maskings:
        dino_imgs = []
        for i, out_i in enumerate(sam_out):
            mask = out_i['segmentation']
            masked_img = np.einsum('ij,cij->cij', mask, img)
            bbox = [int(x) for x in out_i['bbox']]
            masked_img_cropped = masked_img[:, bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            masked_img_cropped_tensor = torch.tensor(masked_img_cropped, dtype=torch.float32, device=self.device) / 255
            dino_img_i = dino_process_img(masked_img_cropped_tensor, has_channels=True,
                                          normalization_tr=fake_normalization)
            dino_imgs.append(dino_img_i)
        dino_imgs = torch.stack(dino_imgs)
        # embed the sam generated masks
        maks_zs = self._embed(dino_imgs)  # (num_masks, embed_dim)
        # compute the scores
        sim_matrix = torch.einsum('rz,mz->rm', self.renered_zs, maks_zs)  # (num_rendered, num_masks)

        mask_max_scores, mask_max_scores_indxs = sim_matrix.max(dim=0)
        # mask_scores = mask_max_scores
        mask_scores = sim_matrix.mean(dim=0)
        topk_scores, topk_mask_indxs = torch.topk(mask_scores, top_k)
        topk_rendered_indxs = mask_max_scores_indxs[topk_mask_indxs]

        best_mask = sam_out[topk_mask_indxs[0]]['segmentation']

        if visualize:
            # create two figures
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            img_reshaped = img.transpose(1, 2, 0) # (h, w, 3)
            ax[0].imshow(img_reshaped)
            ax[1].imshow(img_reshaped)
            ax[2].imshow(best_mask)
            show_anns(sam_out, ax=ax[1])
            plt.axis('off')

            # top k masks
            fig, axes = plt.subplots(top_k, 2)
            for i, (topk_mask_indx_i, topk_rendered_indx_i) in enumerate(zip(topk_mask_indxs, topk_rendered_indxs)):
                ax_i = axes[i]
                ax_i[0].imshow(dino_imgs[topk_mask_indx_i].detach().cpu().numpy().transpose(1, 2, 0))
                ax_i[1].imshow(self.rendered_imgs[topk_rendered_indx_i].detach().cpu().numpy().transpose(1, 2, 0))
            plt.show()

        return best_mask

    def _embed(self, imgs: torch.Tensor) -> torch.Tensor:
        # dino_imgs: (..., 3, h, w)
        rendered_imgs_dino = dino_process_img(imgs, has_channels=True, normalization_tr=fake_normalization) # (..., 3, 224, 224)
        with torch.no_grad():
            zs = self.dino_wrapper.get_features(rendered_imgs_dino)  # (..., feat_size)
        return zs

    def _embed_rendered_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        rendered_imgs = torch.tensor(np.stack([s['rendered_object'] for s in self.object_rendered_dataset], axis=0),
                                     dtype=torch.float32, device=self.device) / 255.
        rendered_zs = self._embed(rendered_imgs)
        return rendered_zs, rendered_imgs

    def _init_sam_generator(self):
        if self.sam_type == 'sam2':
            sam_kwargs = {
                'points_per_side': 32,
                'pred_iou_thresh': 0.86,
                'stability_score_thresh': 0.86,
                'box_nms_thresh': 0.7,
                'crop_n_layers': 1,
                'crop_n_points_downscale_factor': 1,
                # 'min_mask_region_area': 300,
            }
            # sam_generator = SAM2AutomaticMaskGenerator.from_pretrained('facebook/sam2-hiera-large', **sam_kwargs)
            # sam_generator = SAM2AutomaticMaskGenerator.from_pretrained('facebook/sam2-hiera-small', **sam_kwargs)
            # sam2 = build_sam2('sam2_hiera_l.yaml', get_sam_checkpoint('sam2-hiera-large'), device=self.device, apply_postprocessing=False)
            sam_generator = SAM2AutomaticMaskGenerator(
                self.sam_wrapper.sam,
                **sam_kwargs
            )
        else:
            # sam 1
            # sam = sam_model_registry['vit_h'](checkpoint=get_sam_checkpoint()).to(self.device)
            sam = self.sam_wrapper.sam
            sam_generator = SamAutomaticMaskGenerator(sam,
                                                      points_per_side=32,
                                                      pred_iou_thresh=0.86,
                                                      # stability_score_thresh=0.92,
                                                      # stability_score_thresh=0.97,
                                                      stability_score_thresh=0.86,
                                                      box_nms_thresh=0.7,
                                                      crop_n_layers=1,
                                                      # crop_n_layers=2,
                                                      # crop_n_points_downscale_factor=2,
                                                      crop_n_points_downscale_factor=1,
                                                      min_mask_region_area=300,  # Requires open-cv to run post-processing
                                                      # min_mask_region_area=0,  # Requires open-cv to run post-processing
                                                      )
        return sam_generator

    def _init_sam_wrapper(self):
        if self.sam_type == 'sam2':
            sam_wrapper = SAM2Wrapper()
        else:
            # sam 1
            sam_wrapper = SAMWrapper()
        return sam_wrapper

    def _init_dino(self):
        dino_wrapper = DINOWrapper(model_name=self.dino_model_name).to(self.device)
        return dino_wrapper


def filter_small_masks(sam_out, min_mask_size, max_mask_size=None):
    filtered_sam_out = []
    for i, out_i in enumerate(sam_out):
        mask_size_i = out_i['segmentation'].sum()
        if mask_size_i >= min_mask_size:
            if max_mask_size is not None:
                if mask_size_i <= max_mask_size:
                    filtered_sam_out.append(out_i)
            else:
                filtered_sam_out.append(out_i)
    return filtered_sam_out


def fake_normalization(x):
    return x




