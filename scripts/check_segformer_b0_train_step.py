from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from config import (
    BATCH_SIZE,
    SEGFORMER_B0_ENCODER_WEIGHTS,
    SEGFORMER_B0_LEARNING_RATE,
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
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(dataloader))
    images = batch["image"].to(device)
    masks = batch["mask"].to(device)

    model = create_segformer_b0(
        num_classes=RUGD_NUM_CLASSES,
        encoder_weights=SEGFORMER_B0_ENCODER_WEIGHTS,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=SEGFORMER_B0_LEARNING_RATE,
    )

    model.train()
    optimizer.zero_grad()

    first_parameter = next(model.parameters())
    parameter_before = first_parameter.detach().clone()

    logits = model(images)
    loss = criterion(logits, masks)
    loss.backward()

    grad_norm = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            grad_norm += parameter.grad.detach().norm().item()

    optimizer.step()

    parameter_change = (first_parameter.detach() - parameter_before).abs().sum().item()

    print("Checking one training step SegFormer-B0")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Train split: {SEGFORMER_B0_TRAIN_SPLIT_PATH}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Input image shape:  {images.shape}")
    print(f"Input mask shape:   {masks.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Loss value: {loss.item():.4f}")
    print(f"Total grad norm: {grad_norm:.4f}")
    print(f"First parameter absolute change: {parameter_change:.8f}")

    if not torch.isfinite(loss):
        raise RuntimeError("Loss is NaN or Inf")

    if grad_norm <= 0:
        raise RuntimeError("Gradients were not computed")

    if parameter_change <= 0:
        raise RuntimeError("optimizer.step() did not change model parameters")

    print()
    print("One training step completed successfully.")


if __name__ == "__main__":
    main()

