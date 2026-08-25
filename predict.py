import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from src.model import UNet
from src.utils import get_device, overlay_mask, preprocess_grayscale


def load_model(checkpoint, device):
    payload = torch.load(checkpoint, map_location=device)
    model = UNet().to(device)
    model.load_state_dict(payload['model_state_dict'])
    model.eval()
    return model, payload.get('image_size', 256)


def predict_image(model, image, device, size, threshold=0.5):
    tensor, original_size = preprocess_grayscale(image, size)
    with torch.no_grad():
        logits = model(tensor.to(device))
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    prob = cv2.resize(prob, original_size, interpolation=cv2.INTER_LINEAR)
    return (prob >= threshold).astype(np.uint8) * 255, prob


def main(args):
    device = get_device()
    model, size = load_model(args.checkpoint, device)
    image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(args.image)

    mask, _ = predict_image(model, image, device, size, args.threshold)
    overlay = overlay_mask(image, mask)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), overlay)
    cv2.imwrite(str(out.with_name(out.stem + '_mask.png')), mask)
    print(f'Saved overlay to {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--checkpoint', default='checkpoints/unet_best.pt')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output', default='outputs/prediction.png')
    main(parser.parse_args())
