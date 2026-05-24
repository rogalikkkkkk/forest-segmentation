from pathlib import Path
import argparse
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    RUGD_CLASS_WEIGHTS_NPY_PATH,
    RUGD_NUM_CLASSES,
    RUGD_SAMPLE_COLORMAP_PATH,
    SEGFORMER_B0_BEST_CHECKPOINT_PATH,
    SEGFORMER_B0_BEST_METRICS_PATH,
    SEGFORMER_B0_CHECKPOINT_PATH,
    SEGFORMER_B0_ENCODER_WEIGHTS,
    SEGFORMER_B0_HISTORY_PATH,
    SEGFORMER_B0_LEARNING_RATE,
    SEGFORMER_B0_LOG_EVERY_N_BATCHES,
    SEGFORMER_B0_LOSS,
    SEGFORMER_B0_LOSS_HISTORY_PATH,
    SEGFORMER_B0_NUM_EPOCHS,
    SEGFORMER_B0_OUTPUT_DIR,
    SEGFORMER_B0_OPTIMIZER,
    SEGFORMER_B0_SCHEDULER,
    SEGFORMER_B0_STEP_HISTORY_PATH,
    SEGFORMER_B0_STEP_LOSS_CURVE_PATH,
    SEGFORMER_B0_TRAINING_CURVES_PATH,
    SEGFORMER_B0_TRAIN_IMAGES_DIR,
    SEGFORMER_B0_TRAIN_MASKS_ID_DIR,
    SEGFORMER_B0_TRAIN_SPLIT_PATH,
    SEGFORMER_B0_VAL_SPLIT_PATH,
    SEGFORMER_B0_WEIGHT_DECAY,
)
from datasets.rugd_dataset import RUGDDataset
from experiment_utils import copy_existing_artifacts, prepare_run_dir, save_config_snapshot
from losses import SUPPORTED_LOSSES, create_loss
from models.segformer_b0 import create_segformer_b0
from training_utils import (
    create_optimizer,
    create_scheduler,
    get_current_learning_rate,
    step_scheduler,
)
from visualization_utils import (
    read_id_to_color,
    save_epoch_prediction_grid,
    save_selected_samples,
    select_fixed_random_indices,
)


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


def get_class_status(gt_pixels, pred_pixels, true_positive):
    if gt_pixels == 0 and pred_pixels == 0:
        return "absent_in_gt_and_prediction"

    if gt_pixels == 0 and pred_pixels > 0:
        return "absent_in_gt_false_positive"

    if gt_pixels > 0 and pred_pixels == 0:
        return "present_in_gt_not_predicted"

    if true_positive == 0:
        return "present_but_no_true_positive"

    return "present"


def get_class_statistics(confusion_matrix):
    true_positive = np.diag(confusion_matrix)
    ground_truth_pixels = confusion_matrix.sum(axis=1)
    predicted_pixels = confusion_matrix.sum(axis=0)
    union_pixels = ground_truth_pixels + predicted_pixels - true_positive

    return ground_truth_pixels, predicted_pixels, true_positive, union_pixels


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    epoch,
    start_global_step,
    step_history,
):
    model.train()

    total_loss = 0.0
    total_images = 0
    global_step = start_global_step

    for batch_index, batch in enumerate(dataloader, start=1):
        global_step += 1

        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, masks)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss is NaN or Inf on train batch {batch_index}")

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_images += batch_size
        step_history.append(
            {
                "global_step": global_step,
                "epoch": epoch,
                "batch": batch_index,
                "train_loss": loss.item(),
            }
        )

        if (
            SEGFORMER_B0_LOG_EVERY_N_BATCHES > 0
            and batch_index % SEGFORMER_B0_LOG_EVERY_N_BATCHES == 0
        ):
            print(
                f"  train batch {batch_index:04d}/{len(dataloader):04d} "
                f"loss: {loss.item():.4f}"
            )

    return total_loss / total_images, global_step


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_images = 0
    confusion_matrix = np.zeros((RUGD_NUM_CLASSES, RUGD_NUM_CLASSES), dtype=np.int64)

    with torch.inference_mode():
        for batch_index, batch in enumerate(dataloader, start=1):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            loss = criterion(logits, masks)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Loss is NaN or Inf on val batch {batch_index}")

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_images += batch_size

            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            targets = masks.cpu().numpy()

            update_confusion_matrix(
                confusion_matrix=confusion_matrix,
                predictions=predictions,
                targets=targets,
                num_classes=RUGD_NUM_CLASSES,
            )

    val_loss = total_loss / total_images
    pixel_accuracy, mean_iou, iou_per_class = calculate_metrics(confusion_matrix)

    return val_loss, pixel_accuracy, mean_iou, iou_per_class, confusion_matrix


def create_checkpoint(
    model,
    optimizer,
    epoch,
    history,
    best_mean_iou,
    best_epoch,
    iou_per_class,
    num_epochs,
    learning_rate,
    optimizer_name,
    weight_decay,
    scheduler_name,
    loss_name,
):
    return {
        "model_name": "segformer_b0",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "num_classes": RUGD_NUM_CLASSES,
        "image_height": IMAGE_HEIGHT,
        "image_width": IMAGE_WIDTH,
        "batch_size": BATCH_SIZE,
        "encoder_weights": SEGFORMER_B0_ENCODER_WEIGHTS,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "optimizer": optimizer_name,
        "weight_decay": weight_decay,
        "scheduler": scheduler_name,
        "loss": loss_name,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_mean_iou": best_mean_iou,
        "history": history,
        "iou_per_class": iou_per_class.tolist(),
        "train_images_dir": str(SEGFORMER_B0_TRAIN_IMAGES_DIR),
        "train_masks_id_dir": str(SEGFORMER_B0_TRAIN_MASKS_ID_DIR),
        "train_split_path": str(SEGFORMER_B0_TRAIN_SPLIT_PATH),
        "val_split_path": str(SEGFORMER_B0_VAL_SPLIT_PATH),
    }


def save_history(history):
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_pixel_accuracy",
        "val_mean_iou",
        "is_best",
    ]

    with SEGFORMER_B0_HISTORY_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    with SEGFORMER_B0_LOSS_HISTORY_PATH.open("w", encoding="utf-8") as file:
        for row in history:
            file.write(f"{row['epoch']},{row['train_loss']:.6f}\n")


def save_step_history(step_history):
    fieldnames = ["global_step", "epoch", "batch", "train_loss"]

    with SEGFORMER_B0_STEP_HISTORY_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(step_history)


def save_best_metrics_with_class_statistics(
    epoch,
    mean_iou,
    pixel_accuracy,
    val_loss,
    iou_per_class,
    confusion_matrix,
):
    (
        ground_truth_pixels,
        predicted_pixels,
        true_positive,
        union_pixels,
    ) = get_class_statistics(confusion_matrix)

    with SEGFORMER_B0_BEST_METRICS_PATH.open("w", encoding="utf-8") as file:
        file.write("metric,value\n")
        file.write(f"best_epoch,{epoch}\n")
        file.write(f"val_loss,{val_loss:.6f}\n")
        file.write(f"val_pixel_accuracy,{pixel_accuracy:.6f}\n")
        file.write(f"val_mean_iou,{mean_iou:.6f}\n")
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


def plot_training_curves(history):
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    val_mean_iou = [row["val_mean_iou"] for row in history]
    val_pixel_accuracy = [row["val_pixel_accuracy"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, train_loss, marker="o", label="train loss")
    axes[0].plot(epochs, val_loss, marker="o", label="val loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, val_mean_iou, marker="o", label="val mIoU")
    axes[1].plot(epochs, val_pixel_accuracy, marker="o", label="val pixel accuracy")
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric value")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(SEGFORMER_B0_TRAINING_CURVES_PATH, dpi=150)
    plt.close()


def plot_step_loss_curve(step_history):
    if not step_history:
        return

    steps = [row["global_step"] for row in step_history]
    train_losses = [row["train_loss"] for row in step_history]

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.plot(steps, train_losses, linewidth=0.8, alpha=0.8, label="batch train loss")
    ax.set_title("Train loss by optimization step")
    ax.set_xlabel("Global step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(SEGFORMER_B0_STEP_LOSS_CURVE_PATH, dpi=150)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train SegFormer-B0 on RUGD.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=SEGFORMER_B0_NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=SEGFORMER_B0_LEARNING_RATE)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default=SEGFORMER_B0_OPTIMIZER)
    parser.add_argument("--weight-decay", type=float, default=SEGFORMER_B0_WEIGHT_DECAY)
    parser.add_argument("--scheduler", choices=["none", "plateau"], default=SEGFORMER_B0_SCHEDULER)
    parser.add_argument("--loss", choices=SUPPORTED_LOSSES, default=SEGFORMER_B0_LOSS)
    parser.add_argument("--prediction-interval", type=int, default=5)
    parser.add_argument("--prediction-epochs", nargs="+", type=int, default=[1, 5, 10, 15, 20, 25, 30])
    parser.add_argument("--prediction-sample-count", type=int, default=5)
    parser.add_argument("--prediction-seed", type=int, default=42)
    parser.add_argument("--disable-epoch-predictions", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = (
        prepare_run_dir(SEGFORMER_B0_OUTPUT_DIR, run_dir=args.run_dir)
        if args.run_dir is not None
        else None
    )

    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = RUGDDataset(
        images_dir=SEGFORMER_B0_TRAIN_IMAGES_DIR,
        masks_dir=SEGFORMER_B0_TRAIN_MASKS_ID_DIR,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        image_mean=IMAGE_MEAN,
        image_std=IMAGE_STD,
        split_file=SEGFORMER_B0_TRAIN_SPLIT_PATH,
    )

    val_dataset = RUGDDataset(
        images_dir=SEGFORMER_B0_TRAIN_IMAGES_DIR,
        masks_dir=SEGFORMER_B0_TRAIN_MASKS_ID_DIR,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        image_mean=IMAGE_MEAN,
        image_std=IMAGE_STD,
        split_file=SEGFORMER_B0_VAL_SPLIT_PATH,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = create_segformer_b0(
        num_classes=RUGD_NUM_CLASSES,
        encoder_weights=SEGFORMER_B0_ENCODER_WEIGHTS,
    ).to(device)

    criterion = create_loss(
        args.loss,
        class_weights_path=RUGD_CLASS_WEIGHTS_NPY_PATH,
        device=device,
    )
    optimizer = create_optimizer(
        model.parameters(),
        optimizer_name=args.optimizer,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = create_scheduler(optimizer, args.scheduler)

    history = []
    step_history = []
    global_step = 0
    best_mean_iou = -1.0
    best_epoch = 0
    best_iou_per_class = np.full(RUGD_NUM_CLASSES, np.nan, dtype=np.float64)
    epoch_predictions_dir = (
        (run_dir if run_dir is not None else SEGFORMER_B0_OUTPUT_DIR)
        / "epoch_predictions"
    )
    selected_val_indices = []
    id_to_color = None
    prediction_epochs = set(args.prediction_epochs)

    if not args.disable_epoch_predictions:
        selected_val_indices = select_fixed_random_indices(
            dataset_size=len(val_dataset),
            sample_count=args.prediction_sample_count,
            seed=args.prediction_seed,
        )
        save_selected_samples(
            val_dataset,
            selected_val_indices,
            epoch_predictions_dir / "selected_val_samples.csv",
        )
        id_to_color = read_id_to_color(RUGD_SAMPLE_COLORMAP_PATH)

    print("Training SegFormer-B0")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Images dir: {SEGFORMER_B0_TRAIN_IMAGES_DIR}")
    print(f"Masks dir: {SEGFORMER_B0_TRAIN_MASKS_ID_DIR}")
    print(f"Train split: {SEGFORMER_B0_TRAIN_SPLIT_PATH}")
    print(f"Val split: {SEGFORMER_B0_VAL_SPLIT_PATH}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Image size: {IMAGE_HEIGHT} x {IMAGE_WIDTH}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of classes: {RUGD_NUM_CLASSES}")
    print(f"Encoder weights: {SEGFORMER_B0_ENCODER_WEIGHTS}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Scheduler: {args.scheduler}")
    print(f"Loss: {args.loss}")
    if args.loss == "weighted_ce":
        print(f"Class weights: {RUGD_CLASS_WEIGHTS_NPY_PATH}")
    if selected_val_indices:
        print(f"Epoch prediction interval: {args.prediction_interval}")
        print(f"Epoch prediction epochs: {sorted(prediction_epochs)}")
        print(f"Epoch prediction samples: {selected_val_indices}")
        print(f"Epoch predictions dir: {epoch_predictions_dir}")
    print(f"Batch logging interval: {SEGFORMER_B0_LOG_EVERY_N_BATCHES}")
    if run_dir is not None:
        print(f"Run dir: {run_dir}")
    print()

    save_config_snapshot(
        run_dir,
        {
            "model": "segformer_b0",
            "batch_size": BATCH_SIZE,
            "image_height": IMAGE_HEIGHT,
            "image_width": IMAGE_WIDTH,
            "num_classes": RUGD_NUM_CLASSES,
            "encoder_weights": SEGFORMER_B0_ENCODER_WEIGHTS,
            "num_epochs": args.epochs,
            "learning_rate": args.lr,
            "optimizer": args.optimizer,
            "weight_decay": args.weight_decay,
            "scheduler": args.scheduler,
            "loss": args.loss,
            "epoch_prediction_interval": args.prediction_interval,
            "epoch_prediction_epochs": sorted(prediction_epochs),
            "epoch_prediction_sample_count": args.prediction_sample_count,
            "epoch_prediction_seed": args.prediction_seed,
            "epoch_prediction_val_indices": selected_val_indices,
            "train_split": SEGFORMER_B0_TRAIN_SPLIT_PATH,
            "val_split": SEGFORMER_B0_VAL_SPLIT_PATH,
        },
    )

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")

        train_loss, global_step = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            start_global_step=global_step,
            step_history=step_history,
        )

        (
            val_loss,
            val_pixel_accuracy,
            val_mean_iou,
            iou_per_class,
            val_confusion_matrix,
        ) = validate_one_epoch(
            model=model,
            dataloader=val_dataloader,
            criterion=criterion,
            device=device,
        )

        is_best = val_mean_iou > best_mean_iou

        if is_best:
            best_mean_iou = val_mean_iou
            best_epoch = epoch
            best_iou_per_class = iou_per_class

        step_scheduler(scheduler, val_mean_iou)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_pixel_accuracy": val_pixel_accuracy,
            "val_mean_iou": val_mean_iou,
            "is_best": int(is_best),
        }
        history.append(row)

        checkpoint = create_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            history=history,
            best_mean_iou=best_mean_iou,
            best_epoch=best_epoch,
            iou_per_class=iou_per_class,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            optimizer_name=args.optimizer,
            weight_decay=args.weight_decay,
            scheduler_name=args.scheduler,
            loss_name=args.loss,
        )

        torch.save(checkpoint, SEGFORMER_B0_CHECKPOINT_PATH)

        if is_best:
            torch.save(checkpoint, SEGFORMER_B0_BEST_CHECKPOINT_PATH)
            save_best_metrics_with_class_statistics(
                epoch=best_epoch,
                mean_iou=best_mean_iou,
                pixel_accuracy=val_pixel_accuracy,
                val_loss=val_loss,
                iou_per_class=best_iou_per_class,
                confusion_matrix=val_confusion_matrix,
            )

        save_history(history)
        save_step_history(step_history)
        plot_training_curves(history)
        plot_step_loss_curve(step_history)

        if (
            selected_val_indices
            and (
                epoch in prediction_epochs
                or (
                    args.prediction_interval > 0
                    and epoch % args.prediction_interval == 0
                    and not prediction_epochs
                )
            )
        ):
            epoch_prediction_path = epoch_predictions_dir / f"epoch_{epoch:03d}.png"
            save_epoch_prediction_grid(
                model=model,
                dataset=val_dataset,
                indices=selected_val_indices,
                device=device,
                output_path=epoch_prediction_path,
                id_to_color=id_to_color,
                image_mean=IMAGE_MEAN,
                image_std=IMAGE_STD,
                title=f"SegFormer-B0 - epoch {epoch}",
            )

        print(f"Train loss: {train_loss:.4f}")
        print(f"Val loss:   {val_loss:.4f}")
        print(f"Val pixel accuracy: {val_pixel_accuracy:.4f}")
        print(f"Val mIoU: {val_mean_iou:.4f}")
        print(f"Best val mIoU: {best_mean_iou:.4f} at epoch {best_epoch}")
        print(f"Current learning rate: {get_current_learning_rate(optimizer):.8f}")
        if is_best:
            print(f"Best checkpoint updated: {SEGFORMER_B0_BEST_CHECKPOINT_PATH}")
        if (
            selected_val_indices
            and (
                epoch in prediction_epochs
                or (
                    args.prediction_interval > 0
                    and epoch % args.prediction_interval == 0
                    and not prediction_epochs
                )
            )
        ):
            print(f"Epoch predictions saved to: {epoch_prediction_path}")
        print()

    print("Training finished successfully.")
    print(f"Last checkpoint saved to: {SEGFORMER_B0_CHECKPOINT_PATH}")
    print(f"Best checkpoint saved to: {SEGFORMER_B0_BEST_CHECKPOINT_PATH}")
    print(f"History saved to: {SEGFORMER_B0_HISTORY_PATH}")
    print(f"Step history saved to: {SEGFORMER_B0_STEP_HISTORY_PATH}")
    print(f"Loss history saved to: {SEGFORMER_B0_LOSS_HISTORY_PATH}")
    print(f"Best metrics saved to: {SEGFORMER_B0_BEST_METRICS_PATH}")
    print(f"Training curves saved to: {SEGFORMER_B0_TRAINING_CURVES_PATH}")
    print(f"Step loss curve saved to: {SEGFORMER_B0_STEP_LOSS_CURVE_PATH}")
    if selected_val_indices:
        print(f"Epoch predictions saved to: {epoch_predictions_dir}")

    copy_existing_artifacts(
        [
            SEGFORMER_B0_CHECKPOINT_PATH,
            SEGFORMER_B0_BEST_CHECKPOINT_PATH,
            SEGFORMER_B0_HISTORY_PATH,
            SEGFORMER_B0_STEP_HISTORY_PATH,
            SEGFORMER_B0_LOSS_HISTORY_PATH,
            SEGFORMER_B0_BEST_METRICS_PATH,
            SEGFORMER_B0_TRAINING_CURVES_PATH,
            SEGFORMER_B0_STEP_LOSS_CURVE_PATH,
        ],
        run_dir,
    )
    if run_dir is not None:
        print(f"Run artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()


