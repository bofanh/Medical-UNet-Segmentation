"""Training utilities for UNet segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader


def dice_coefficient(preds: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    preds = preds.float()
    targets = targets.float()
    intersection = torch.sum(preds * targets)
    union = torch.sum(preds) + torch.sum(targets)
    dice = (2 * intersection + epsilon) / (union + epsilon)
    return dice


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, torch.Tensor]:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    dice = dice_coefficient(preds, targets)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    return {"dice": dice, "bce": bce}


def save_checkpoint(state: Dict[str, Any], checkpoint_dir: Path, is_best: bool, keep_last: int = 3) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    epoch = state.get("epoch", 0)
    path = checkpoint_dir / f"model_epoch_{epoch}.pt"
    torch.save(state, path)

    if is_best:
        best_path = checkpoint_dir / "best_model.pt"
        torch.save(state, best_path)

    checkpoints = sorted(checkpoint_dir.glob("model_epoch_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old_ckpt in checkpoints[keep_last:]:
        old_ckpt.unlink(missing_ok=True)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    amp: bool = True,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_dice = 0.0

    for batch in dataloader:
        images = batch["image"].to(device)
        targets = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            logits = model(images)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        metrics = compute_metrics(logits.detach(), targets)
        running_loss += loss.item() * images.size(0)
        running_dice += metrics["dice"].item() * images.size(0)

    dataset_size = len(dataloader.dataset)
    return running_loss / dataset_size, running_dice / dataset_size


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device, amp: bool = True) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_dice = 0.0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            targets = batch["mask"].to(device)
            with autocast(enabled=amp):
                logits = model(images)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
            metrics = compute_metrics(logits, targets)
            running_loss += loss.item() * images.size(0)
            running_dice += metrics["dice"].item() * images.size(0)

    dataset_size = len(dataloader.dataset)
    return running_loss / dataset_size, running_dice / dataset_size
