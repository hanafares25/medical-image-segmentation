import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.dataset import SpleenSliceDataset
from src.metrics import dice_score, iou_score
from src.model import UNet
from src.utils import get_device


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            probs.sum() + targets.sum() + self.smooth
        )

        return 1.0 - dice


def train(args):
    device = get_device()
    print(f"Using device: {device}")

    # Dataset without augmentation first
    full_dataset = SpleenSliceDataset(
        image_dir=args.images,
        mask_dir=args.masks,
        image_size=args.image_size,
        augment=False,
        skip_empty=True,
    )

    print(f"Total slices: {len(full_dataset)}")

    # Quick sanity check
    image, mask = full_dataset[0]

    print("Sample image shape:", image.shape)
    print("Sample mask shape:", mask.shape)
    print("Image range:", image.min().item(), image.max().item())
    print("Mask values:", mask.unique())

    # Split train / validation
    val_size = max(1, int(len(full_dataset) * args.val_split))
    train_size = len(full_dataset) - val_size

    train_ds, val_ds = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Enable augmentation only for training
    train_ds.dataset.augment = True

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = UNet().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
    )

    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()

    best_dice = -1.0

    Path(args.checkpoint).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(1, args.epochs + 1):

        # -------------------------
        # Training
        # -------------------------
        model.train()

        running_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}'):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            logits = model(images)

            loss = (
                0.5 * bce_loss(logits, masks)
                + 0.5 * dice_loss(logits, masks)
            )

            loss.backward()
            optimizer.step()

            running_loss += (
                loss.item() * images.size(0)
            )

        # -------------------------
        # Validation
        # -------------------------
        model.eval()

        val_dice = 0.0
        val_iou = 0.0

        with torch.no_grad():

            for images, masks in val_loader:

                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)

                val_dice += (
                    dice_score(logits, masks).item()
                    * images.size(0)
                )

                val_iou += (
                    iou_score(logits, masks).item()
                    * images.size(0)
                )

        train_loss = (
            running_loss / max(train_size, 1)
        )

        val_dice /= max(val_size, 1)
        val_iou /= max(val_size, 1)

        print(
            f"Epoch {epoch}: "
            f"loss={train_loss:.4f} "
            f"dice={val_dice:.4f} "
            f"iou={val_iou:.4f}"
        )

        # Save best model
        if val_dice > best_dice:

            best_dice = val_dice

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_size": args.image_size,
                    "best_dice": best_dice,
                },
                args.checkpoint,
            )

            print(
                f"Saved best checkpoint -> "
                f"{args.checkpoint}"
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train U-Net for MSD spleen CT segmentation."
    )

    parser.add_argument(
        "--images",
        default="data/Task09_Spleen/imagesTr",
    )

    parser.add_argument(
        "--masks",
        default="data/Task09_Spleen/labelsTr",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/unet_best.pt",
    )

    train(parser.parse_args())