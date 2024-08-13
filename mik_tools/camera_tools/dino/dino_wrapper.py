import torch
import torch.nn as nn

from mik_tools.mik_learning_tools import batched_img_method_decorator
from .dino_utils import dino_process_img, dino_descriptor_sizes


class DINOWrapper(nn.Module):

    def __init__(self, model_name='dinov2_vitl14', freeze=True):
        super().__init__()
        self.model_name = model_name
        self.freeze = freeze
        assert self.model_name in dino_descriptor_sizes.keys(), f'model_name {self.model_name} not availble. Choose from {dino_descriptor_sizes.keys()}'
        self.num_features = dino_descriptor_sizes[self.model_name]
        self.dino_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=self.model_name)
        if self.freeze:
            self.dino_model.eval()

    def forward(self, *args, **kwargs):
        return self.dino_model(*args, **kwargs)

    @batched_img_method_decorator
    def get_features_raw(self, img: torch.Tensor) -> torch.Tensor:
        """
        Embed the images into latent features
        Args:
            img (torch.Tensor): of shape (..., 3, 224, 224)
        Returns:
            features (torch.Tensor): of shape (..., num_features)
        """
        features = self.dino_model(img)
        return features

    def get_features(self, img: torch.Tensor) -> torch.Tensor:
        """
        Embed the images into latent features
        Args:
            img (torch.Tensor): of shape (..., num_channels, w, h)
        Returns:
            features (torch.Tensor): of shape (..., num_features)

        NOTE: It turns out that a DINO call consumes a lot of memory. If gradients are not needed,
        then calls to this method should be wrapped into a no_grad context:
            with torch.no_grad():
                features = dino_wrapper.get_features(img)
        """
        img_proc = dino_process_img(img, has_channels=True)
        features = self.get_features_raw(img_proc)
        return features