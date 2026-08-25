import numpy as np
import cv2


def window_ct(image, min_hu=-100, max_hu=300):
    """
    Apply CT windowing suitable for abdominal soft tissue.
    """
    image = np.clip(image, min_hu, max_hu)

    image = (image - min_hu) / (max_hu - min_hu)

    return image.astype(np.float32)


def resize_image_and_mask(image, mask, size=(256, 256)):
    image = cv2.resize(
        image,
        size,
        interpolation=cv2.INTER_LINEAR
    )

    mask = cv2.resize(
        mask,
        size,
        interpolation=cv2.INTER_NEAREST
    )

    mask = (mask > 0).astype(np.float32)

    return image, mask