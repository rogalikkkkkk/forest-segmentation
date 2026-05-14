from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from config import (
    IMAGE_HEIGHT,
    IMAGE_MEAN,
    IMAGE_STD,
    IMAGE_WIDTH,
    RUGD_NUM_CLASSES,
    RUGD_SAMPLE_IMAGES_DIR,
    RUGD_SAMPLE_MASKS_ID_DIR,
    RUGD_SAMPLE_COLORMAP_PATH,
    UNET_RESNET34_SAMPLE_CHECKPOINT_PATH,
    UNET_RESNET34_SAMPLE_VISUALIZATION_INDEX,
    UNET_RESNET34_SAMPLE_VISUALIZATION_PATH,
)
from datasets.rugd_dataset import RUGDDataset
from models.unet_resnet34 import UNetResNet34


def read_id_to_color(path):
    id_to_color = {}

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()
            class_id = int(parts[0])
            rgb = tuple(map(int, parts[-3:]))
            id_to_color[class_id] = rgb

    return id_to_color


def denormalize_image(image_tensor):
    image = image_tensor.detach().cpu().numpy()

    mean = np.array(IMAGE_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(IMAGE_STD, dtype=np.float32).reshape(3, 1, 1)

    image = image * std + mean
    image = np.clip(image, 0.0, 1.0)
    image = np.transpose(image, (1, 2, 0))

    return image


def colorize_mask(mask, id_to_color):
    height, width = mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for class_id, color in id_to_color.items():
        color_mask[mask == class_id] = color

    return color_mask


def main():
    if not UNET_RESNET34_SAMPLE_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {UNET_RESNET34_SAMPLE_CHECKPOINT_PATH}. "
            "Run scripts/train_unet_resnet34_sample.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = RUGDDataset(
        images_dir=RUGD_SAMPLE_IMAGES_DIR,
        masks_dir=RUGD_SAMPLE_MASKS_ID_DIR,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        image_mean=IMAGE_MEAN,
        image_std=IMAGE_STD,
    )

    sample = dataset[UNET_RESNET34_SAMPLE_VISUALIZATION_INDEX]
    image = sample["image"].unsqueeze(0).to(device)
    mask = sample["mask"].numpy()
    filename = sample["filename"]

    model = UNetResNet34(
        num_classes=RUGD_NUM_CLASSES,
        encoder_weights=None,
    ).to(device)

    checkpoint = torch.load(UNET_RESNET34_SAMPLE_CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(image)
        prediction = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

    id_to_color = read_id_to_color(RUGD_SAMPLE_COLORMAP_PATH)

    image_for_plot = denormalize_image(sample["image"])
    mask_color = colorize_mask(mask, id_to_color)
    prediction_color = colorize_mask(prediction, id_to_color)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_for_plot)
    axes[0].set_title(f"Image: {filename}")
    axes[0].axis("off")

    axes[1].imshow(mask_color)
    axes[1].set_title("Ground truth")
    axes[1].axis("off")

    axes[2].imshow(prediction_color)
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(UNET_RESNET34_SAMPLE_VISUALIZATION_PATH, dpi=150)
    plt.close()

    print("U-Net ResNet34 prediction visualization")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Checkpoint: {UNET_RESNET34_SAMPLE_CHECKPOINT_PATH}")
    print(f"Sample index: {UNET_RESNET34_SAMPLE_VISUALIZATION_INDEX}")
    print(f"Filename: {filename}")
    print(f"Input shape: {image.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Prediction shape: {prediction.shape}")
    print(f"Ground truth classes: {np.unique(mask).tolist()}")
    print(f"Prediction classes:   {np.unique(prediction).tolist()}")
    print(f"Visualization saved to: {UNET_RESNET34_SAMPLE_VISUALIZATION_PATH}")


if __name__ == "__main__":
    main()
