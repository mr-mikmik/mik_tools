import torch
import torch.nn.functional as F
from torchvision import transforms


def resnet_process_img(img: torch.Tensor, has_channels: bool=False, normalization_tr=None) -> torch.Tensor:
    """
    Transform an image to a desired Resnet format.
    ResNet expects a RGB 256x256 image in a normalized range.
    Args:
        img (...,W, H): image to be converted to resnet format
        has_channels (bool): whether the image has channels. If true, img: (..., num_channels, W, H)
        normalization_tr (): if provided, we will use that normalization
    Returns:
        img_proc (..., 3, 256, 256)
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
        img = img.unsqueeze(-3) # (..., 1, W, H)
    img_flat = img.flatten(end_dim=-4)  # (N, num_channels, W, H)
    img_flat = F.interpolate(img_flat, size=(256, 256), mode='bilinear')  # (..., num_channels, 256, 256)
    if num_channels == 1:
        img_flat = img_flat.repeat_interleave(3, dim=-3)  # (N, 3, W, H)
    elif num_channels == 3:
        pass
    else:
        raise NotImplementedError(f'img_flat has {num_channels} channels (shape: {img_flat.shape}) which is not supported. Either must have 1 or 3 channels.')
    img = img_flat.reshape(batch_dims + (3, 256, 256)) # (..., 3, W, H)
    img_proc = normalization_tr(img)
    return img_proc


# TEST:
if __name__ == '__main__':
    img1 = torch.rand((10, 10, 240, 300))
    img1_proc = resnet_process_img(img1)
    print(img1.shape, '->', img1_proc.shape)

    img2 = torch.rand((10, 1, 240, 300))
    img2_proc = resnet_process_img(img2, has_channels=True)
    print(img2.shape, '->', img2_proc.shape)

    img3 = torch.rand((10, 3, 400, 440))
    img3_proc = resnet_process_img(img3, has_channels=True)
    print(img3.shape, '->', img3_proc.shape)

    normalization_tr = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img4 = torch.rand((10, 3, 400, 440))
    img4_proc = resnet_process_img(img4, has_channels=True, normalization_tr=normalization_tr)
    print(img4.shape, '->', img4_proc.shape)