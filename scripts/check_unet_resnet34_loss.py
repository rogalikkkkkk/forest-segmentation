from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from config import (
    BATCH_SIZE,
    IMAGE_HEIGHT,
    IMAGE_MEAN,
    IMAGE_STD,
    IMAGE_WIDTH,
    RUGD_NUM_CLASSES,
    UNET_RESNET34_ENCODER_WEIGHTS,
    RUGD_SAMPLE_IMAGES_DIR,
    RUGD_SAMPLE_MASKS_ID_DIR,
)
from datasets.rugd_dataset import RUGDDataset
from models.unet_resnet34 import UNetResNet34


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = RUGDDataset(
        images_dir=RUGD_SAMPLE_IMAGES_DIR,
        masks_dir=RUGD_SAMPLE_MASKS_ID_DIR,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        image_mean=IMAGE_MEAN,
        image_std=IMAGE_STD,
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

    model = UNetResNet34(
        num_classes=RUGD_NUM_CLASSES,
        encoder_weights=UNET_RESNET34_ENCODER_WEIGHTS,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    model.train()

    logits = model(images)
    loss = criterion(logits, masks)

    print("Проверка CrossEntropyLoss для U-Net ResNet34")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Logits shape: {logits.shape}")
    print(f"Masks shape:  {masks.shape}")
    print(f"Loss value:   {loss.item():.4f}")

    if not torch.isfinite(loss):
        raise RuntimeError("Loss получился NaN или Inf")

    print()
    print("Loss успешно посчитан.")


if __name__ == "__main__":
    main()
