from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing import window_ct, resize_image_and_mask


class SpleenSliceDataset(Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir,
        image_size=256,
        augment=False,
        skip_empty=True,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.image_size = image_size
        self.augment = augment
        self.skip_empty = skip_empty

        self.samples = []

        image_files = sorted(p for p in self.image_dir.glob("*.nii.gz")if not p.name.startswith("._"))

        print(f"Found {len(image_files)} CT volumes.")

        for image_path in image_files:

            mask_path = self.mask_dir / image_path.name

            if not mask_path.exists():
                print(f"Warning: no mask found for {image_path.name}")
                continue

            image_volume = nib.load(str(image_path))
            mask_volume = nib.load(str(mask_path))

            image_shape = image_volume.shape
            mask_shape = mask_volume.shape

            if image_shape != mask_shape:
                print(
                    f"Skipping {image_path.name}: "
                    f"image shape {image_shape} != mask shape {mask_shape}"
                )
                continue

            mask_data = mask_volume.get_fdata()

            num_slices = image_shape[2]

            for slice_idx in range(num_slices):

                if self.skip_empty:
                    mask_slice = mask_data[:, :, slice_idx]

                    if np.sum(mask_slice) == 0:
                        continue

                self.samples.append(
                    {
                        "image": image_path,
                        "mask": mask_path,
                        "slice_idx": slice_idx,
                    }
                )

        print(f"Using {len(self.samples)} CT slices.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        image_volume = nib.load(
            str(sample["image"])
        ).get_fdata()

        mask_volume = nib.load(
            str(sample["mask"])
        ).get_fdata()

        slice_idx = sample["slice_idx"]

        image = image_volume[:, :, slice_idx]
        mask = mask_volume[:, :, slice_idx]

        # CT normalization/windowing
        image = window_ct(image)

        image, mask = resize_image_and_mask(
            image,
            mask,
            size=(self.image_size, self.image_size)
        )

        if self.augment:

            # Horizontal flip
            if np.random.rand() > 0.5:
                image = np.fliplr(image)
                mask = np.fliplr(mask)

            # Vertical flip
            if np.random.rand() > 0.5:
                image = np.flipud(image)
                mask = np.flipud(mask)

            # Small rotation
            if np.random.rand() > 0.5:
                k = np.random.randint(1, 4)

                image = np.rot90(image, k)
                mask = np.rot90(mask, k)

        # np.flip / np.rot90 can produce negative strides
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)

        image = torch.tensor(
            image,
            dtype=torch.float32
        ).unsqueeze(0)

        mask = torch.tensor(
            mask,
            dtype=torch.float32
        ).unsqueeze(0)

        return image, mask