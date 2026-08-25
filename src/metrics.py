import torch


def dice_score(logits, targets, threshold: float = 0.5, eps: float = 1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = targets.float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return ((2 * intersection + eps) / (union + eps)).mean()


def iou_score(logits, targets, threshold: float = 0.5, eps: float = 1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = targets.float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    return ((intersection + eps) / (union + eps)).mean()
