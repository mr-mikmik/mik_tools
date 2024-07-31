import torch
import torch.nn.functional as F
from torchvision import transforms


dino_descriptor_sizes = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


def dino_process_img(img: torch.Tensor, has_channels: bool=False, normalization_tr=None, dino_img_size=224) -> torch.Tensor:
    """
    Transform an image to a desired Resnet format.
    ResNet expects a RGB 256x256 image in a normalized range.
    Args:
        img (...,W, H): image to be converted to resnet format
        has_channels (bool): whether the image has channels. If true, img: (..., num_channels, W, H)
        normalization_tr (): if provided, we will use that normalization
    Returns:
        img_proc (..., 3, dino_img_size, dino_img_size)
    """
    img_shape = img.shape
    if normalization_tr is None:
        normalization_tr = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if has_channels:
        num_channels = img_shape[-3]
        batch_dims = img_shape[:-3]
    else:
        num_channels = 1
        batch_dims = img_shape[:-2]
        img = img.unsqueeze(-3)  # (..., 1, W, H)
    img_flat = img.flatten(end_dim=-4)  # (N, num_channels, W, H)
    img_flat = F.interpolate(img_flat, size=(dino_img_size, dino_img_size), mode='bilinear')  # (..., num_channels, 256, 256)
    if num_channels == 1:
        img_flat = img_flat.repeat_interleave(3, dim=-3)  # (N, 3, W, H)
    elif num_channels == 3:
        pass
    else:
        raise NotImplementedError(
            f'img_flat has {num_channels} channels (shape: {img_flat.shape}) which is not supported. Either must have 1 or 3 channels.')
    img = img_flat.reshape(batch_dims + (3, dino_img_size, dino_img_size))  # (..., 3, W, H)
    img_proc = normalization_tr(img)
    return img_proc