"""Training entrypoint for the medical UNet project."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import MedicalImageDataset, default_pair_transform
from src.models.unet import UNet
from src.utils.config import load_config
from src.utils.training import evaluate, log_training_run, save_checkpoint, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a UNet on medical image segmentation data")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloader(
    root: Path,
    images_subdir: str,
    masks_subdir: str,
    batch_size: int,
    num_workers: int,
    transform,
    shuffle: bool,
) -> DataLoader:
    dataset = MedicalImageDataset(
        image_dir=root / images_subdir,
        mask_dir=root / masks_subdir,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def resolve_scheduler(optimizer: torch.optim.Optimizer, config: dict) -> torch.optim.lr_scheduler._LRScheduler | None:
    sched_type = config.get("type")
    if sched_type is None:
        return None

    if sched_type == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=config.get("factor", 0.5),
            patience=config.get("patience", 5),
            threshold=config.get("threshold", 0.01),
            min_lr=config.get("min_lr", 1e-6),
        )
    raise ValueError(f"Unsupported scheduler type: {sched_type}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed = config.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths = config.get("paths", {})
    dataset_root = Path(paths.get("dataset_root", ".")).expanduser().resolve()

    training_cfg = config.get("training", {})
    batch_size = int(training_cfg.get("batch_size", 2))
    num_workers = int(training_cfg.get("num_workers", 4))
    epochs = int(training_cfg.get("epochs", 1))
    amp = bool(training_cfg.get("amp", True))
    lr = float(training_cfg.get("learning_rate", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 1e-5))

    model_cfg = config.get("model", {})
    model = UNet(
        input_channels=int(model_cfg.get("input_channels", 1)),
        num_classes=int(model_cfg.get("num_classes", 1)),
        base_channels=int(model_cfg.get("base_channels", 64)),
    ).to(device)

    transform = default_pair_transform()

    train_loader = build_dataloader(
        dataset_root,
        paths.get("train_images", "images/train"),
        paths.get("train_masks", "masks/train"),
        batch_size,
        num_workers,
        transform,
        shuffle=True,
    )

    val_loader = build_dataloader(
        dataset_root,
        paths.get("val_images", "images/val"),
        paths.get("val_masks", "masks/val"),
        batch_size,
        num_workers,
        transform,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler_cfg = config.get("scheduler", {})
    scheduler = resolve_scheduler(optimizer, scheduler_cfg)

    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    output_dir = Path(paths.get("output_dir", "outputs"))
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    best_dice = -float("inf")
    best_val_loss = float("inf")
    keep_last = int(config.get("checkpoints", {}).get("keep_last", 3))
    save_best_only = bool(config.get("checkpoints", {}).get("save_best_only", True))
    last_train_loss = None
    last_train_dice = None
    last_val_loss = None
    last_val_dice = None

    for epoch in range(1, epochs + 1):
        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, device, scaler, amp=amp)
        val_loss, val_dice = evaluate(model, val_loader, device, amp=amp)
        last_train_loss = train_loss
        last_train_dice = train_dice
        last_val_loss = val_loss
        last_val_dice = val_dice

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_dice)
        elif scheduler is not None:
            scheduler.step()

        is_best = val_dice > best_dice
        if is_best:
            best_dice = val_dice
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "val_dice": val_dice,
            "val_loss": val_loss,
        }

        if is_best or not save_best_only:
            save_checkpoint(state, checkpoints_dir, is_best=is_best, keep_last=keep_last)

        print(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}"
        )

    print(f"Training complete. Best Dice: {best_dice:.4f}")
    run_summary = {
        "config_path": str(Path(args.config).resolve()),
        "epochs": epochs,
        "best_dice": float(best_dice) if best_dice != -float("inf") else None,
        "best_val_loss": float(best_val_loss) if best_val_loss != float("inf") else None,
        "final_train_loss": float(last_train_loss) if last_train_loss is not None else None,
        "final_val_loss": float(last_val_loss) if last_val_loss is not None else None,
        "final_train_dice": float(last_train_dice) if last_train_dice is not None else None,
        "final_val_dice": float(last_val_dice) if last_val_dice is not None else None,
        "seed": seed,
        "config": config,
    }
    run_dir = log_training_run(Path(args.config), output_dir, run_summary)
    print(f"Run metadata saved to {run_dir}")


if __name__ == "__main__":
    main()
