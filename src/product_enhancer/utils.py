import torch
from PIL import ImageDraw
import numpy as np


def combine_masks(labels, masks):
    unique_labels = set(labels)
    combined_masks = []
    label_to_index = {}
    idx2label = {}
    for i, label in enumerate(unique_labels):
        label_masks = [mask for l, mask in zip(labels, masks) if l == label]
        combined_mask = torch.stack(label_masks).sum(dim=0) > 0
        combined_masks.append(combined_mask)
        label_to_index[label] = i
        idx2label[i] = label
    combined_mask = torch.stack(combined_masks)
    mask_numpy = combined_mask.numpy().astype(np.uint8) * 255
    mask_numpy = mask_numpy.transpose(0, 2, 3, 1)
    gray_array = np.mean(mask_numpy, axis=3, dtype=np.uint8)
    binary_mask = np.where(gray_array > 0, 255, 0).astype(np.uint8)
    return binary_mask, label_to_index, idx2label


def draw_box_on_image(image, top_left_x, top_left_y, bottom_right_x, bottom_right_y):
    """
    Draw a bounding box on the image.

    Args:
    image (PIL.Image.Image): The input image.
    top_left_x (int): The x-coordinate of the top-left corner of the bounding box.
    top_left_y (int): The y-coordinate of the top-left corner of the bounding box.
    bottom_right_x (int): The x-coordinate of the bottom-right corner of the bounding box.
    bottom_right_y (int): The y-coordinate of the bottom-right corner of the bounding box.

    Returns:
    PIL.Image.Image: The image with the bounding box drawn on it.
    """
    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Draw the bounding box
    draw.rectangle([top_left_x, top_left_y, bottom_right_x, bottom_right_y], outline="red", width=3)

    return image
