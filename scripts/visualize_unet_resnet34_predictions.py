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
    UNET_RESNET34_BEST_CHECKPOINT_PATH,
    UNET_RESNET34_ENCODER_WEIGHTS,
    UNET_RESNET34_PREDICTIONS_GRID_PATH,
    UNET_RESNET34_VISUALIZATION_INDICES,
)
from datasets.rugd_dataset import RUGDDataset
from experiment_utils import get_run_artifact_path
from models.unet_resnet34 import UNetResNet34
from visualization_utils import colorize_mask, denormalize_image, read_id_to_color


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize U-Net ResNet34 predictions.")
    parser.add_argument("--run-dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir

    checkpoint_path = get_run_artifact_path(UNET_RESNET34_BEST_CHECKPOINT_PATH, run_dir)
    predictions_grid_path = get_run_artifact_path(
        UNET_RESNET34_PREDICTIONS_GRID_PATH,
        run_dir,
    )

    if not checkpoint_path.exists():
        if run_dir is not None:
            raise FileNotFoundError(
                f"Best checkpoint not found in run directory: {checkpoint_path}. "
                "Run scripts/train_unet_resnet34.py with the same --run-dir first."
            )
        raise FileNotFoundError(
            f"Best checkpoint not found: {checkpoint_path}. "
            "Run scripts/train_unet_resnet34.py first."
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

    model = UNetResNet34(
        num_classes=RUGD_NUM_CLASSES,
        encoder_weights=UNET_RESNET34_ENCODER_WEIGHTS,
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
        index for index in UNET_RESNET34_VISUALIZATION_INDICES if index < len(dataset)
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

            image_for_plot = denormalize_image(sample["image"], IMAGE_MEAN, IMAGE_STD)
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

    predictions_grid_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(predictions_grid_path, dpi=150)
    plt.close()

    print("U-Net ResNet34 predictions visualization")
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
