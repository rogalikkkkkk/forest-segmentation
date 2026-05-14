from pathlib import Path
import csv
import sys

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from config import (
    RUGD_CLASS_WEIGHTS_CSV_PATH,
    RUGD_CLASS_WEIGHTS_NPY_PATH,
    RUGD_MASKS_ID_DIR,
    RUGD_NUM_CLASSES,
    RUGD_TRAIN_SPLIT_PATH,
)


def read_split_names(split_path):
    with Path(split_path).open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def find_mask_path(mask_name):
    direct_path = RUGD_MASKS_ID_DIR / mask_name
    if direct_path.exists():
        return direct_path

    stem_path = RUGD_MASKS_ID_DIR / f"{Path(mask_name).stem}.png"
    if stem_path.exists():
        return stem_path

    raise FileNotFoundError(f"Mask not found for split item: {mask_name}")


def compute_class_counts(mask_names):
    counts = np.zeros(RUGD_NUM_CLASSES, dtype=np.int64)

    for index, mask_name in enumerate(mask_names, start=1):
        mask_path = find_mask_path(mask_name)
        mask = np.array(Image.open(mask_path), dtype=np.int64)

        if mask.ndim != 2:
            raise ValueError(f"Expected 2D ID mask, got shape {mask.shape}: {mask_path}")

        valid_pixels = (mask >= 0) & (mask < RUGD_NUM_CLASSES)
        counts += np.bincount(
            mask[valid_pixels].ravel(),
            minlength=RUGD_NUM_CLASSES,
        )

        if index % 500 == 0:
            print(f"Processed masks: {index}/{len(mask_names)}")

    return counts


def compute_enet_weights(class_counts):
    total_pixels = class_counts.sum()
    if total_pixels == 0:
        raise ValueError("No pixels found in train masks.")

    frequencies = class_counts.astype(np.float64) / total_pixels
    weights = np.zeros(RUGD_NUM_CLASSES, dtype=np.float64)
    present_classes = frequencies > 0

    weights[present_classes] = 1.0 / np.log(1.02 + frequencies[present_classes])
    weights[present_classes] /= weights[present_classes].mean()

    return frequencies, weights.astype(np.float32)


def save_weights(class_counts, frequencies, weights):
    RUGD_CLASS_WEIGHTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.save(RUGD_CLASS_WEIGHTS_NPY_PATH, weights)

    with RUGD_CLASS_WEIGHTS_CSV_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["class_id", "pixels", "frequency", "weight"])

        for class_id in range(RUGD_NUM_CLASSES):
            writer.writerow(
                [
                    class_id,
                    int(class_counts[class_id]),
                    f"{frequencies[class_id]:.12f}",
                    f"{weights[class_id]:.8f}",
                ]
            )


def main():
    mask_names = read_split_names(RUGD_TRAIN_SPLIT_PATH)

    print("Computing RUGD class weights from train split")
    print("=" * 60)
    print(f"Train split: {RUGD_TRAIN_SPLIT_PATH}")
    print(f"Masks dir:   {RUGD_MASKS_ID_DIR}")
    print(f"Masks:       {len(mask_names)}")

    class_counts = compute_class_counts(mask_names)
    frequencies, weights = compute_enet_weights(class_counts)
    save_weights(class_counts, frequencies, weights)

    print()
    print(f"Weights CSV saved to: {RUGD_CLASS_WEIGHTS_CSV_PATH}")
    print(f"Weights NPY saved to: {RUGD_CLASS_WEIGHTS_NPY_PATH}")
    print()
    print("class_id,pixels,frequency,weight")
    for class_id in range(RUGD_NUM_CLASSES):
        print(
            f"{class_id},{int(class_counts[class_id])},"
            f"{frequencies[class_id]:.8f},{weights[class_id]:.6f}"
        )


if __name__ == "__main__":
    main()
