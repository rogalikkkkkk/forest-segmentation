from pathlib import Path
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
    RUGD_NUM_CLASSES,
    RUGD_SAMPLE_IMAGES_DIR,
    RUGD_SAMPLE_MASKS_ID_DIR,
    UNET_RESNET34_SAMPLE_CHECKPOINT_PATH,
    UNET_RESNET34_SAMPLE_METRICS_PATH,
    UNET_RESNET34_SAMPLE_SAVE_PER_IMAGE_METRICS,
)
from datasets.rugd_dataset import RUGDDataset
from models.unet_resnet34 import UNetResNet34


def synchronize_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def update_confusion_matrix(confusion_matrix, predictions, targets, num_classes):
    valid_pixels = (targets >= 0) & (targets < num_classes)

    encoded = num_classes * targets[valid_pixels] + predictions[valid_pixels]
    batch_confusion = np.bincount(
        encoded,
        minlength=num_classes * num_classes,
    )
    batch_confusion = batch_confusion.reshape(num_classes, num_classes)

    confusion_matrix += batch_confusion


def calculate_metrics(confusion_matrix):
    true_positive = np.diag(confusion_matrix)
    ground_truth_pixels = confusion_matrix.sum(axis=1)
    predicted_pixels = confusion_matrix.sum(axis=0)

    total_correct = true_positive.sum()
    total_pixels = confusion_matrix.sum()
    pixel_accuracy = total_correct / total_pixels

    union = ground_truth_pixels + predicted_pixels - true_positive
    valid_classes = union > 0
    iou_per_class = np.full(confusion_matrix.shape[0], np.nan, dtype=np.float64)
    iou_per_class[valid_classes] = true_positive[valid_classes] / union[valid_classes]
    mean_iou = np.nanmean(iou_per_class)

    return pixel_accuracy, mean_iou, iou_per_class


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

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = UNetResNet34(
        num_classes=RUGD_NUM_CLASSES,
        encoder_weights=None,
    ).to(device)

    checkpoint = torch.load(UNET_RESNET34_SAMPLE_CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    confusion_matrix = np.zeros((RUGD_NUM_CLASSES, RUGD_NUM_CLASSES), dtype=np.int64)
    per_image_metrics = []
    per_image_processing_times = []

    with torch.no_grad():
        for batch in dataloader:
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

            for filename, prediction, mask in zip(filenames, predictions, masks):
                pixel_accuracy = (prediction == mask).mean()
                per_image_metrics.append((filename, pixel_accuracy))

            update_confusion_matrix(
                confusion_matrix=confusion_matrix,
                predictions=predictions,
                targets=masks,
                num_classes=RUGD_NUM_CLASSES,
            )

    average_processing_time = float(
        np.mean([time_value for _, time_value in per_image_processing_times])
    )
    pixel_accuracy, mean_iou, iou_per_class = calculate_metrics(confusion_matrix)

    print("Evaluation U-Net ResNet34 on RUGD sample")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Checkpoint: {UNET_RESNET34_SAMPLE_CHECKPOINT_PATH}")
    print(f"Pixel Accuracy: {pixel_accuracy:.4f}")
    print(f"Mean IoU:        {mean_iou:.4f}")
    print(
        "Average processing time per image: "
        f"{average_processing_time:.6f} s "
        f"({average_processing_time * 1000:.3f} ms)"
    )
    print()
    print("Processing time per image:")
    for filename, processing_time in per_image_processing_times:
        print(f"  {filename}: {processing_time:.6f} s ({processing_time * 1000:.3f} ms)")
    if UNET_RESNET34_SAMPLE_SAVE_PER_IMAGE_METRICS:
        print()
        print("Pixel Accuracy per image:")
        for filename, image_pixel_accuracy in per_image_metrics:
            print(f"  {filename}: {image_pixel_accuracy:.4f}")
    print()
    print("IoU per class:")
    for class_id, class_iou in enumerate(iou_per_class):
        if np.isnan(class_iou):
            print(f"  class {class_id:02d}: n/a")
        else:
            print(f"  class {class_id:02d}: {class_iou:.4f}")

    with UNET_RESNET34_SAMPLE_METRICS_PATH.open("w", encoding="utf-8") as file:
        file.write("metric,value\n")
        file.write(f"pixel_accuracy,{pixel_accuracy:.6f}\n")
        file.write(f"mean_iou,{mean_iou:.6f}\n")
        file.write(f"average_processing_time_seconds,{average_processing_time:.6f}\n")
        file.write(f"average_processing_time_ms,{average_processing_time * 1000:.3f}\n")
        file.write("\nfilename,processing_time_seconds,processing_time_ms\n")
        for filename, processing_time in per_image_processing_times:
            file.write(
                f"{filename},{processing_time:.6f},{processing_time * 1000:.3f}\n"
            )
        if UNET_RESNET34_SAMPLE_SAVE_PER_IMAGE_METRICS:
            file.write("\nfilename,pixel_accuracy\n")
            for filename, image_pixel_accuracy in per_image_metrics:
                file.write(f"{filename},{image_pixel_accuracy:.6f}\n")
        file.write("\nclass_id,iou\n")
        for class_id, class_iou in enumerate(iou_per_class):
            if np.isnan(class_iou):
                file.write(f"{class_id},nan\n")
            else:
                file.write(f"{class_id},{class_iou:.6f}\n")

    print()
    print(f"Metrics saved to: {UNET_RESNET34_SAMPLE_METRICS_PATH}")


if __name__ == "__main__":
    main()
