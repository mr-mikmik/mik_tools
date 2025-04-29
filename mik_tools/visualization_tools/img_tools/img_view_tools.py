import numpy as np


def apply_alpha_shading(img: np.ndarray,
                        mask: np.ndarray,
                        alpha: float = 0.5,
                        shade_color: tuple = (0, 0, 0)
                        ) -> np.ndarray:
    """
    Return a new image where pixels _outside_ the mask are blended with shade_color.

    Parameters
    ----------
    img : np.ndarray, shape (..., H, W, 3), dtype uint8
        Original RGB image.
    mask : np.ndarray, shape (..., H, W), dtype bool or 0/1
        Boolean mask. True = keep original, False = apply shading.
    alpha : float in [0,1]
        Blending factor (0=no shading, 1=full shade_color).
    shade_color : 3-tuple of ints
        RGB color to blend in where mask is False.
    """
    # make sure types are correct
    # convert to float for blending
    out = img.astype(np.float32)
    shade = np.array(shade_color, dtype=np.float32)

    # indices to shade
    inv = ~mask.astype(bool)
    # blend: out = out*(1-alpha) + shade*alpha
    out[inv] = out[inv] * (1.0 - alpha) + shade * alpha

    # clip and convert back
    return np.clip(out, 0, 255).astype(np.uint8)