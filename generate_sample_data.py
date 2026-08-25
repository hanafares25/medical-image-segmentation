"""Generate a tiny synthetic dataset so the project works without private medical data."""
from pathlib import Path
import random

import cv2
import numpy as np

IMAGE_DIR = Path('data/images')
MASK_DIR = Path('data/masks')
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)
np.random.seed(42)

for i in range(80):
    size = 256
    image = np.random.normal(70, 18, (size, size)).clip(0, 255).astype(np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)

    cx = random.randint(70, 186)
    cy = random.randint(70, 186)
    rx = random.randint(20, 55)
    ry = random.randint(18, 48)
    angle = random.randint(0, 180)

    cv2.ellipse(mask, (cx, cy), (rx, ry), angle, 0, 360, 255, -1)

    lesion_texture = np.random.normal(random.randint(135, 190), 12, (size, size)).clip(0, 255).astype(np.uint8)
    image = np.where(mask > 0, lesion_texture, image).astype(np.uint8)
    image = cv2.GaussianBlur(image, (5, 5), 0)

    name = f'sample_{i:03d}.png'
    cv2.imwrite(str(IMAGE_DIR / name), image)
    cv2.imwrite(str(MASK_DIR / name), mask)

print(f'Generated 80 synthetic image/mask pairs in {IMAGE_DIR} and {MASK_DIR}.')
