"""Inference script for UNet medical segmentation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from PIL import Image
import torch
from torchvision import transforms as T

from src.models.unet import UNet
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained UNet model")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained checkpoint")
    parser.add_argument("--image", type=str, help="Single image to segment")
    parser.add_argument("--image-dir", type=str, help="Directory of images to segment")
    parser.add_argument("--output", type=str, required=True, help="Output file or directory for masks")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for binarization")
    return parser.parse_args()


def load_model(config_path: Path, checkpoint_path: Path, device: torch.device) -> UNet:
    config = load_config(config_path)
    model_cfg = config.get("model", {})
    model = UNet(
        input_channels=int(model_cfg.get("input_channels", 1)),
        num_classes=int(model_cfg.get("num_classes", 1)),
        base_channels=int(model_cfg.get("base_channels", 64)),
    )
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()
    return model


def gather_images(image_path: str | None, image_dir: str | None) -> Iterable[Path]:
    if image_path is None and image_dir is None:
        raise ValueError("Provide either --image or --image-dir")

    if image_path is not None:
        yield Path(image_path)
        return

    for path in sorted(Path(image_dir).iterdir()):
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            yield path


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = Path(args.config).expanduser()
    checkpoint_path = Path(args.checkpoint).expanduser()
    output_path = Path(args.output).expanduser()

    model = load_model(config_path, checkpoint_path, device)

    preprocess = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    if args.image_dir:
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for image_path in gather_images(args.image, args.image_dir):
            image = Image.open(image_path)
            image = image.convert("L") if model.input_channels == 1 else image.convert("RGB")
            tensor = preprocess(image).unsqueeze(0).to(device)
            logits = model(tensor)
            probs = torch.sigmoid(logits)
            mask = (probs > args.threshold).float().squeeze(0).squeeze(0)
            mask_img = Image.fromarray((mask.cpu().numpy() * 255).astype("uint8"))

            if args.image_dir:
                output_file = output_path / f"{image_path.stem}_mask.png"
            else:
                output_file = output_path

            mask_img.save(output_file)
            print(f"Saved segmentation to {output_file}")


if __name__ == "__main__":
    main()
