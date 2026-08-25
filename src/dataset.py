from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class MedicalSegmentationDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, image_size: int = 256):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.images = sorted([p for p in self.image_dir.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}])

        if not self.images:
            raise ValueError(f"No images found in {self.image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.mask_dir / image_path.name

        if not mask_path.exists():
            raise FileNotFoundError(f"Matching mask not found for {image_path.name}")

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        mask = (mask.astype(np.float32) / 255.0 > 0.5).astype(np.float32)

        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return image, mask, image_path.name
