# Medical UNet Segmentation

A lightweight PyTorch project for training a UNet model on 2D medical image segmentation tasks.

## Features
- Modular UNet implementation with configurable channel sizes and optional bilinear upsampling.
- Paired image/mask dataset loader with simple joint transforms.
- Config-driven training loop with AMP support, Dice/BCE metrics, and checkpoint rotation.
- Inference script for batch or single-image segmentation.

## Setup

```bash
conda activate medical-unet-seg
pip install -r requirements.txt
```

## Training

```bash
python train.py --config configs/train.yaml
```

Expected dataset layout (update paths in the config accordingly):
```
/your/dataset/root
├── images
│   ├── train
│   │   ├── case1.png
│   │   └── ...
│   └── val
│       ├── caseX.png
│       └── ...
└── masks
    ├── train
    │   ├── case1.png
    │   └── ...
    └── val
        ├── caseX.png
        └── ...
```

## Inference

```bash
python infer.py --config configs/train.yaml --checkpoint outputs/checkpoints/best_model.pt --image /path/to/image.png --output ./mask.png
# or batch mode
python infer.py --config configs/train.yaml --checkpoint outputs/checkpoints/best_model.pt --image-dir /path/to/images --output ./pred_masks
```

## Notes
- Adjust normalization/augmentations via `src/data/dataset.py` or wrap with `torchvision` transforms.
- For multi-class or 3D volumes, extend the dataset and UNet accordingly.
- Use `configs/train.yaml` as a template for additional experiments.
