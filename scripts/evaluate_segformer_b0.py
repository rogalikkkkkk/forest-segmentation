from pathlib import Path
import argparse
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from config import (
    BATCH_SIZE,
    IMAGE_HEIGHT,
    IMAGE_MEAN,
    IMAGE_STD,
    IMAGE_WIDTH,
    RUGD_IMAGES_DIR,
    RUGD_MASKS_ID_DIR,
    RUGD_NUM_CLASSES,
    RUGD_TEST_SPLIT_PATH,
    SEGFORMER_B0_BEST_CHECKPOINT_PATH,
    SEGFORMER_B0_ENCODER_WEIGHTS,
    SEGFORMER_B0_METRICS_PATH,
)
from datasets.rugd_dataset import RUGDDataset
from experiment_utils import get_run_artifact_path
from models.segformer_b0 import create_segformer_b0
from segmentation_metrics import (
    calculate_metrics,
    get_class_status,
    update_confusion_matrix,
)


def synchronize_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def write_metrics_file(
    path,
    pixel_accuracy,
    mean_iou,
    confusion_matrix,
    iou_per_class,
    average_processing_time,
    per_image_processing_times,
):
    true_positive = np.diag(confusion_matrix)
    ground_truth_pixels = confusion_matrix.sum(axis=1)
    predicted_pixels = confusion_matrix.sum(axis=0)
    union_pixels = ground_truth_pixels + predicted_pixels - true_positive

    with path.open("w", encoding="utf-8") as file:
        file.write("metric,value\n")
        file.write(f"pixel_accuracy,{pixel_accuracy:.6f}\n")
        file.write(f"mean_iou,{mean_iou:.6f}\n")
        file.write(f"average_processing_time_seconds,{average_processing_time:.6f}\n")
        file.write(f"average_processing_time_ms,{average_processing_time * 1000:.3f}\n")
        file.write("\n")
        file.write("filename,processing_time_seconds,processing_time_ms\n")
        for filename, processing_time in per_image_processing_times:
            file.write(
                f"{filename},{processing_time:.6f},{processing_time * 1000:.3f}\n"
            )
        file.write("\n")
        file.write(
            "class_id,gt_pixels,pred_pixels,tp_pixels,union_pixels,iou,status\n"
        )

        for class_id, class_iou in enumerate(iou_per_class):
            gt_pixels = int(ground_truth_pixels[class_id])
            pred_pixels = int(predicted_pixels[class_id])
            tp_pixels = int(true_positive[class_id])
            union = int(union_pixels[class_id])
            status = get_class_status(gt_pixels, pred_pixels, tp_pixels)

            if np.isnan(class_iou):
                iou_value = "nan"
            else:
                iou_value = f"{class_iou:.6f}"

            file.write(
                f"{class_id},{gt_pixels},{pred_pixels},{tp_pixels},"
                f"{union},{iou_value},{status}\n"
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SegFormer-B0 on RUGD test split.")
    parser.add_argument("--run-dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = get_run_artifact_path(SEGFORMER_B0_BEST_CHECKPOINT_PATH, run_dir)
    if run_dir is not None and not checkpoint_path.exists():
        checkpoint_path = SEGFORMER_B0_BEST_CHECKPOINT_PATH
    metrics_path = get_run_artifact_path(SEGFORMER_B0_METRICS_PATH, run_dir)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found: {checkpoint_path}. "
            "Run scripts/train_segformer_b0.py first."
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

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = create_segformer_b0(
        num_classes=RUGD_NUM_CLASSES,
        encoder_weights=SEGFORMER_B0_ENCODER_WEIGHTS,
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    confusion_matrix = np.zeros((RUGD_NUM_CLASSES, RUGD_NUM_CLASSES), dtype=np.int64)
    per_image_processing_times = []

    print("Evaluation SegFormer-B0")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Images dir: {RUGD_IMAGES_DIR}")
    print(f"Masks dir: {RUGD_MASKS_ID_DIR}")
    print(f"Test split: {RUGD_TEST_SPLIT_PATH}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Best checkpoint: {checkpoint_path}")
    print()

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader, start=1):
            images = batch["image"].to(device)
            masks = batch["mask"].cpu().numpy()
            filenames = batch["filename"]

            synchronize_device(device)
            start_time = time.perf_counter()
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            synchronize_device(device)
            batch_processing_time = time.perf_counter() - start_time
            image_processing_time = batch_processing_time / images.size(0)
            per_image_processing_times.extend(
                (filename, image_processing_time) for filename in filenames
            )

            predictions = predictions.cpu().numpy()

            update_confusion_matrix(
                confusion_matrix=confusion_matrix,
                predictions=predictions,
                targets=masks,
                num_classes=RUGD_NUM_CLASSES,
            )

            if batch_index % 500 == 0:
                print(f"  evaluated batch {batch_index:04d}/{len(dataloader):04d}")

    average_processing_time = float(
        np.mean([time_value for _, time_value in per_image_processing_times])
    )
    pixel_accuracy, mean_iou, iou_per_class = calculate_metrics(confusion_matrix)

    print()
    print(f"Pixel Accuracy: {pixel_accuracy:.4f}")
    print(f"Mean IoU:        {mean_iou:.4f}")
    print(
        "Average processing time per image: "
        f"{average_processing_time:.6f} s "
        f"({average_processing_time * 1000:.3f} ms)"
    )
    print()
    print("IoU per class:")
    for class_id, class_iou in enumerate(iou_per_class):
        if np.isnan(class_iou):
            print(f"  class {class_id:02d}: n/a")
        else:
            print(f"  class {class_id:02d}: {class_iou:.4f}")

    write_metrics_file(
        path=metrics_path,
        pixel_accuracy=pixel_accuracy,
        mean_iou=mean_iou,
        confusion_matrix=confusion_matrix,
        iou_per_class=iou_per_class,
        average_processing_time=average_processing_time,
        per_image_processing_times=per_image_processing_times,
    )

    print()
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()


