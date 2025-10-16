"""Dataset utilities for paired image/mask segmentation datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


class MedicalImageDataset(Dataset):
    """Loads paired image and mask files from two directories."""

    def __init__(
        self,
        image_dir: Path | str,
        mask_dir: Path | str,
        transform: Optional[Callable[[Image.Image, Image.Image], Tuple[torch.Tensor, torch.Tensor]]] = None,
        image_suffixes: Sequence[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
        grayscale: bool = True,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.grayscale = grayscale

        if not self.image_dir.exists() or not self.mask_dir.exists():
            raise FileNotFoundError("Image or mask directory does not exist")

        self.samples = self._discover_samples(image_suffixes)
        if not self.samples:
            raise RuntimeError(f"No image files with suffix {image_suffixes} were found.")

        self.default_transform = T.Compose([T.ToTensor()])

    def _discover_samples(self, suffixes: Sequence[str]) -> list[tuple[Path, Path]]:
        samples: list[tuple[Path, Path]] = []
        for img_file in sorted(self.image_dir.iterdir()):
            if img_file.suffix.lower() not in suffixes:
                continue
            mask_file = self.mask_dir / img_file.name
            if mask_file.exists():
                samples.append((img_file, mask_file))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image_path, mask_path = self.samples[idx]
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        if self.grayscale:
            image = image.convert("L")
            mask = mask.convert("L")
        else:
            image = image.convert("RGB")
            mask = mask.convert("L")

        if self.transform is not None:
            image_tensor, mask_tensor = self.transform(image, mask)
        else:
            image_tensor = self.default_transform(image)
            mask_tensor = self.default_transform(mask)

        mask_tensor = (mask_tensor > 0.5).float()

        return {"image": image_tensor, "mask": mask_tensor, "image_path": str(image_path)}


def default_pair_transform(resize: Optional[int] = None, normalize: bool = True) -> Callable:
    """Creates a simple joint transform for image/mask pairs."""

    image_transforms = []
    mask_transforms = []

    if resize is not None:
        image_transforms.append(T.Resize(resize))
        mask_transforms.append(T.Resize(resize))

    image_transforms.append(T.ToTensor())
    mask_transforms.append(T.ToTensor())

    if normalize:
        image_transforms.append(T.Normalize(mean=[0.5], std=[0.5]))

    image_pipeline = T.Compose(image_transforms)
    mask_pipeline = T.Compose(mask_transforms)

    def apply(image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        return image_pipeline(image), mask_pipeline(mask)

    return apply
