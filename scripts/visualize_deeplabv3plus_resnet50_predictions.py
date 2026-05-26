from pathlib import Path
import argparse
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
    RUGD_COLORMAP_PATH,
    RUGD_IMAGES_DIR,
    RUGD_MASKS_ID_DIR,
    RUGD_NUM_CLASSES,
    RUGD_TEST_SPLIT_PATH,
    DEEPLABV3PLUS_RESNET50_BEST_CHECKPOINT_PATH,
    DEEPLABV3PLUS_RESNET50_ENCODER_WEIGHTS,
    DEEPLABV3PLUS_RESNET50_PREDICTIONS_GRID_PATH,
    DEEPLABV3PLUS_RESNET50_VISUALIZATION_INDICES,
)
from datasets.rugd_dataset import RUGDDataset
from experiment_utils import get_run_artifact_path
from models.deeplabv3plus_resnet50 import create_deeplabv3plus_resnet50


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


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize DeepLabV3+ ResNet50 predictions.")
    parser.add_argument("--run-dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = get_run_artifact_path(
        DEEPLABV3PLUS_RESNET50_BEST_CHECKPOINT_PATH,
        run_dir,
    )
    if run_dir is not None and not checkpoint_path.exists():
        checkpoint_path = DEEPLABV3PLUS_RESNET50_BEST_CHECKPOINT_PATH
    predictions_grid_path = get_run_artifact_path(
        DEEPLABV3PLUS_RESNET50_PREDICTIONS_GRID_PATH,
        run_dir,
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found: {checkpoint_path}. "
            "Run scripts/train_deeplabv3plus_resnet50.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = RUGDDataset(
        images_dir=RUGD_IMAGES_DIR,
        masks_dir=RUGD_MASKS_ID_DIR,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        image_mean=IMAGE_MEAN,
        image_std=IMAGE_STD,
        split_file=RUGD_TEST_SPLIT_PATH,
    )

    model = create_deeplabv3plus_resnet50(
        num_classes=RUGD_NUM_CLASSES,
        encoder_weights=DEEPLABV3PLUS_RESNET50_ENCODER_WEIGHTS,
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    id_to_color = read_id_to_color(RUGD_COLORMAP_PATH)

    valid_indices = [
        index for index in DEEPLABV3PLUS_RESNET50_VISUALIZATION_INDICES if index < len(dataset)
    ]

    if not valid_indices:
        raise RuntimeError("No valid visualization indices for the current dataset.")

    fig, axes = plt.subplots(
        len(valid_indices),
        3,
        figsize=(12, 4 * len(valid_indices)),
    )

    if len(valid_indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    rows_info = []

    with torch.no_grad():
        for row, sample_index in enumerate(valid_indices):
            sample = dataset[sample_index]

            image = sample["image"].unsqueeze(0).to(device)
            mask = sample["mask"].numpy()
            filename = sample["filename"]

            logits = model(image)
            prediction = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

            image_for_plot = denormalize_image(sample["image"])
            mask_color = colorize_mask(mask, id_to_color)
            prediction_color = colorize_mask(prediction, id_to_color)

            axes[row, 0].imshow(image_for_plot)
            axes[row, 0].set_title(f"Image\n{filename}")
            axes[row, 0].axis("off")

            axes[row, 1].imshow(mask_color)
            axes[row, 1].set_title("Ground truth")
            axes[row, 1].axis("off")

            axes[row, 2].imshow(prediction_color)
            axes[row, 2].set_title("Prediction")
            axes[row, 2].axis("off")

            pixel_accuracy = (prediction == mask).mean()
            rows_info.append((sample_index, filename, pixel_accuracy))

    plt.tight_layout()
    plt.savefig(predictions_grid_path, dpi=150)
    plt.close()

    print("DeepLabV3+ ResNet50 predictions visualization")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Test split: {RUGD_TEST_SPLIT_PATH}")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Saved to: {predictions_grid_path}")
    print()
    print("Visualized samples:")
    for sample_index, filename, pixel_accuracy in rows_info:
        print(f"  index {sample_index:04d}: {filename}, pixel accuracy {pixel_accuracy:.4f}")


if __name__ == "__main__":
    main()
