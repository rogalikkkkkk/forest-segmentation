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
    RUGD_SAMPLE_IMAGES_DIR,
    RUGD_SAMPLE_MASKS_ID_DIR,
    UNET_RESNET34_SAMPLE_CHECKPOINT_PATH,
    UNET_RESNET34_SAMPLE_LEARNING_RATE,
    UNET_RESNET34_SAMPLE_LOSS_HISTORY_PATH,
    UNET_RESNET34_SAMPLE_LOG_EVERY_N_BATCHES,
    UNET_RESNET34_SAMPLE_NUM_EPOCHS,
)
from datasets.rugd_dataset import RUGDDataset
from models.unet_resnet34 import UNetResNet34


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_images = 0

    for batch_index, batch in enumerate(dataloader, start=1):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, masks)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss is NaN or Inf on batch {batch_index}")

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_images += batch_size

        if (
            UNET_RESNET34_SAMPLE_LOG_EVERY_N_BATCHES > 0
            and batch_index % UNET_RESNET34_SAMPLE_LOG_EVERY_N_BATCHES == 0
        ):
            print(
                f"  batch {batch_index:02d}/{len(dataloader):02d} "
                f"loss: {loss.item():.4f}"
            )

    return total_loss / total_images


def main():
    torch.manual_seed(42)

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
        shuffle=True,
        num_workers=0,
    )

    model = UNetResNet34(
        num_classes=RUGD_NUM_CLASSES,
        encoder_weights=None,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=UNET_RESNET34_SAMPLE_LEARNING_RATE,
    )

    loss_history = []

    print("Training U-Net ResNet34 on RUGD sample")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Image size: {IMAGE_HEIGHT} x {IMAGE_WIDTH}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of classes: {RUGD_NUM_CLASSES}")
    print(f"Epochs: {UNET_RESNET34_SAMPLE_NUM_EPOCHS}")
    print(f"Learning rate: {UNET_RESNET34_SAMPLE_LEARNING_RATE}")
    print(f"Batch logging interval: {UNET_RESNET34_SAMPLE_LOG_EVERY_N_BATCHES}")
    print()

    for epoch in range(1, UNET_RESNET34_SAMPLE_NUM_EPOCHS + 1):
        print(f"Epoch {epoch}/{UNET_RESNET34_SAMPLE_NUM_EPOCHS}")
        average_loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        loss_history.append(average_loss)
        print(f"Average train loss: {average_loss:.4f}")
        print()

    torch.save(
        {
            "model_name": "unet_resnet34",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "num_classes": RUGD_NUM_CLASSES,
            "image_height": IMAGE_HEIGHT,
            "image_width": IMAGE_WIDTH,
            "batch_size": BATCH_SIZE,
            "num_epochs": UNET_RESNET34_SAMPLE_NUM_EPOCHS,
            "learning_rate": UNET_RESNET34_SAMPLE_LEARNING_RATE,
            "loss_history": loss_history,
        },
        UNET_RESNET34_SAMPLE_CHECKPOINT_PATH,
    )

    with UNET_RESNET34_SAMPLE_LOSS_HISTORY_PATH.open("w", encoding="utf-8") as file:
        for epoch, loss in enumerate(loss_history, start=1):
            file.write(f"{epoch},{loss:.6f}\n")

    print("Training check finished successfully.")
    print(f"Checkpoint saved to: {UNET_RESNET34_SAMPLE_CHECKPOINT_PATH}")
    print(f"Loss history saved to: {UNET_RESNET34_SAMPLE_LOSS_HISTORY_PATH}")


if __name__ == "__main__":
    main()
