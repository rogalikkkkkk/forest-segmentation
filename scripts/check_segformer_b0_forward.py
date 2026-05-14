from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from config import (
    BATCH_SIZE,
    SEGFORMER_B0_ENCODER_WEIGHTS,
    IMAGE_HEIGHT,
    IMAGE_MEAN,
    IMAGE_STD,
    IMAGE_WIDTH,
    RUGD_NUM_CLASSES,
    SEGFORMER_B0_TRAIN_IMAGES_DIR,
    SEGFORMER_B0_TRAIN_MASKS_ID_DIR,
    SEGFORMER_B0_TRAIN_SPLIT_PATH,
)
from datasets.rugd_dataset import RUGDDataset
from models.segformer_b0 import create_segformer_b0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = RUGDDataset(
        images_dir=SEGFORMER_B0_TRAIN_IMAGES_DIR,
        masks_dir=SEGFORMER_B0_TRAIN_MASKS_ID_DIR,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        image_mean=IMAGE_MEAN,
        image_std=IMAGE_STD,
        split_file=SEGFORMER_B0_TRAIN_SPLIT_PATH,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(dataloader))
    images = batch["image"].to(device)
    masks = batch["mask"].to(device)

    model = create_segformer_b0(
        num_classes=RUGD_NUM_CLASSES,
        encoder_weights=SEGFORMER_B0_ENCODER_WEIGHTS,
    ).to(device)
    model.eval()

    with torch.no_grad():
        logits = model(images)

    actual_batch_size = images.size(0)
    expected_shape = (
        actual_batch_size,
        RUGD_NUM_CLASSES,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
    )

    print("Checking forward pass SegFormer-B0")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Train split: {SEGFORMER_B0_TRAIN_SPLIT_PATH}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Configured batch size: {BATCH_SIZE}")
    print(f"Actual batch size: {actual_batch_size}")
    print(f"Input image shape: {images.shape}")
    print(f"Input mask shape:  {masks.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape:      {expected_shape}")
    print(f"Output dtype: {logits.dtype}")
    print(f"Output min/max: {logits.min().item():.4f} / {logits.max().item():.4f}")

    if tuple(logits.shape) != expected_shape:
        raise RuntimeError(
            f"Wrong model output shape: {tuple(logits.shape)} != {expected_shape}"
        )

    print()
    print("Forward pass completed successfully.")


if __name__ == "__main__":
    main()

