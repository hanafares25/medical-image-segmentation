import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.dataset import MedicalSegmentationDataset
from src.metrics import dice_score, iou_score
from src.model import UNet
from src.utils import get_device


def train(args):
    device = get_device()
    print(f'Using device: {device}')

    dataset = MedicalSegmentationDataset(args.images, args.masks, args.image_size)
    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_dice = -1.0
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, masks, _ in tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}'):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        model.eval()
        val_dice = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                val_dice += dice_score(logits, masks).item() * images.size(0)
                val_iou += iou_score(logits, masks).item() * images.size(0)

        train_loss = running_loss / max(train_size, 1)
        val_dice /= max(val_size, 1)
        val_iou /= max(val_size, 1)
        print(f'Epoch {epoch}: loss={train_loss:.4f} dice={val_dice:.4f} iou={val_iou:.4f}')

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({'model_state_dict': model.state_dict(), 'image_size': args.image_size}, args.checkpoint)
            print(f'Saved best checkpoint -> {args.checkpoint}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train U-Net for binary medical image segmentation.')
    parser.add_argument('--images', default='data/images')
    parser.add_argument('--masks', default='data/masks')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--checkpoint', default='checkpoints/unet_best.pt')
    train(parser.parse_args())
