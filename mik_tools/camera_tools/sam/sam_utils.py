import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from mik_tools.aux.package_utils import MODELS_PATH


def get_sam_checkpoint():
    model_name = 'sam_vit_h_4b8939'
    model_path = os.path.join(MODELS_PATH, 'sam', f'{model_name}.pth')
    return model_path


def load_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def show_mask(mask, ax, random_color=False, color=None):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    elif color is not None:
        alpha = 0.5
        color = np.concatenate([np.array(color), np.array([alpha])], axis=0)
    else:
        # color = np.array([30/255, 144/255, 255/255, 0.6])
        color = np.array([30 / 255, 255 / 255, 144 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def plot_sam_masks(img, point_coords, point_labels, masks, scores, logits):
    fig, axes = plt.subplots(1, len(masks), figsize=(5 * len(masks), 5))

    for i, (mask, score) in enumerate(zip(masks, scores)):
        ax = axes[i]
        ax.imshow(img)
        show_mask(mask, ax, random_color=False)
        show_points(point_coords, point_labels, ax)
        ax.set_title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        ax.axis('off')

    return axes