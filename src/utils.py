from pathlib import Path

import cv2
import numpy as np
import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def preprocess_grayscale(image, size=256):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    original_size = (image.shape[1], image.shape[0])
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    return tensor, original_size


def overlay_mask(image, mask, alpha=0.4):
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay = image.copy()
    overlay[mask > 0] = (0, 0, 255)
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
