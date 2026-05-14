from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from config import (
    BATCH_SIZE,
    DEEPLABV3PLUS_RESNET50_ENCODER_WEIGHTS,
    IMAGE_HEIGHT,
    IMAGE_MEAN,
    IMAGE_STD,
    IMAGE_WIDTH,
    RUGD_NUM_CLASSES,
    DEEPLABV3PLUS_RESNET50_TRAIN_IMAGES_DIR,
    DEEPLABV3PLUS_RESNET50_TRAIN_MASKS_ID_DIR,
    DEEPLABV3PLUS_RESNET50_TRAIN_SPLIT_PATH,
)
from datasets.rugd_dataset import RUGDDataset
from models.deeplabv3plus_resnet50 import create_deeplabv3plus_resnet50


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = RUGDDataset(
        images_dir=DEEPLABV3PLUS_RESNET50_TRAIN_IMAGES_DIR,
        masks_dir=DEEPLABV3PLUS_RESNET50_TRAIN_MASKS_ID_DIR,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        image_mean=IMAGE_MEAN,
        image_std=IMAGE_STD,
        split_file=DEEPLABV3PLUS_RESNET50_TRAIN_SPLIT_PATH,
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

    model = create_deeplabv3plus_resnet50(
        num_classes=RUGD_NUM_CLASSES,
        encoder_weights=DEEPLABV3PLUS_RESNET50_ENCODER_WEIGHTS,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    model.train()
    logits = model(images)
    loss = criterion(logits, masks)

    print("Checking CrossEntropyLoss DeepLabV3+ ResNet50")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Train split: {DEEPLABV3PLUS_RESNET50_TRAIN_SPLIT_PATH}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Logits shape: {logits.shape}")
    print(f"Masks shape:  {masks.shape}")
    print(f"Loss value:   {loss.item():.4f}")

    if not torch.isfinite(loss):
        raise RuntimeError("Loss is NaN or Inf")

    print()
    print("Loss computed successfully.")


if __name__ == "__main__":
    main()
