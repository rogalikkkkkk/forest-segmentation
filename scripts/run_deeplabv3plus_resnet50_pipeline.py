from pathlib import Path
import argparse
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from config import (
    EXPERIMENTS_SUMMARY_PATH,
    DEEPLABV3PLUS_RESNET50_BEST_CHECKPOINT_PATH,
    DEEPLABV3PLUS_RESNET50_BEST_METRICS_PATH,
    DEEPLABV3PLUS_RESNET50_CHECKPOINT_PATH,
    DEEPLABV3PLUS_RESNET50_HISTORY_PATH,
    DEEPLABV3PLUS_RESNET50_LOSS_HISTORY_PATH,
    DEEPLABV3PLUS_RESNET50_METRICS_PATH,
    DEEPLABV3PLUS_RESNET50_OUTPUT_DIR,
    DEEPLABV3PLUS_RESNET50_LEARNING_RATE,
    DEEPLABV3PLUS_RESNET50_LOSS,
    DEEPLABV3PLUS_RESNET50_NUM_EPOCHS,
    DEEPLABV3PLUS_RESNET50_OPTIMIZER,
    DEEPLABV3PLUS_RESNET50_PREDICTIONS_GRID_PATH,
    DEEPLABV3PLUS_RESNET50_SCHEDULER,
    DEEPLABV3PLUS_RESNET50_STEP_HISTORY_PATH,
    DEEPLABV3PLUS_RESNET50_STEP_LOSS_CURVE_PATH,
    DEEPLABV3PLUS_RESNET50_TRAINING_CURVES_PATH,
    DEEPLABV3PLUS_RESNET50_WEIGHT_DECAY,
    RUGD_TEST_SPLIT_PATH,
    RUGD_TRAIN_SPLIT_PATH,
    RUGD_VAL_SPLIT_PATH,
)
from experiment_utils import append_experiment_summary, prepare_run_dir, read_metric_value


PIPELINE_STEPS = {
    "prepare_full": "prepare_rugd_full.py",
    "create_splits": "create_rugd_splits.py",
    "check_dataloader": "check_rugd_dataloader.py",
    "check_forward": "check_deeplabv3plus_resnet50_forward.py",
    "check_loss": "check_deeplabv3plus_resnet50_loss.py",
    "check_train_step": "check_deeplabv3plus_resnet50_train_step.py",
    "train": "train_deeplabv3plus_resnet50.py",
    "evaluate": "evaluate_deeplabv3plus_resnet50.py",
    "visualize": "visualize_deeplabv3plus_resnet50_predictions.py",
}


MAIN_PIPELINE = [
    "create_splits",
    "train",
    "evaluate",
    "visualize",
]

FULL_PIPELINE = [
    "prepare_full",
    *MAIN_PIPELINE,
]

CHECK_PIPELINE = [
    "check_forward",
    "check_loss",
    "check_train_step",
]


RUN_AWARE_STEPS = {"train", "evaluate", "visualize"}


def run_script(step_name, run_dir=None, train_args=None):
    script_name = PIPELINE_STEPS[step_name]
    script_path = SCRIPTS_DIR / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    print()
    print("=" * 80)
    print(f"Running step: {step_name}")
    print(f"Script: {script_path}")
    print("=" * 80)

    command = [sys.executable, str(script_path)]
    if run_dir is not None and step_name in RUN_AWARE_STEPS:
        command.extend(["--run-dir", str(run_dir)])
    if step_name == "train" and train_args is not None:
        command.extend(train_args)

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


def collect_steps(args):
    steps = []

    if args.full:
        steps.extend(FULL_PIPELINE)

    if args.main:
        steps.extend(MAIN_PIPELINE)

    if args.checks:
        steps.extend(CHECK_PIPELINE)

    if args.prepare_full:
        steps.append("prepare_full")

    if args.create_splits:
        steps.append("create_splits")

    if args.train:
        steps.append("train")

    if args.evaluate:
        steps.append("evaluate")

    if args.visualize:
        steps.append("visualize")

    if args.check_dataloader:
        steps.append("check_dataloader")

    if args.check_forward:
        steps.append("check_forward")

    if args.check_loss:
        steps.append("check_loss")

    if args.check_train_step:
        steps.append("check_train_step")

    ordered_unique_steps = []
    seen = set()

    for step in steps:
        if step not in seen:
            ordered_unique_steps.append(step)
            seen.add(step)

    return ordered_unique_steps


def append_run_summary(run_dir, args):
    if run_dir is None:
        return

    best_metrics_path = run_dir / DEEPLABV3PLUS_RESNET50_BEST_METRICS_PATH.name
    metrics_path = run_dir / DEEPLABV3PLUS_RESNET50_METRICS_PATH.name

    append_experiment_summary(
        EXPERIMENTS_SUMMARY_PATH,
        {
            "run_id": run_dir.name,
            "model": "deeplabv3plus_resnet50",
            "run_dir": run_dir,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "optimizer": args.optimizer,
            "weight_decay": args.weight_decay,
            "scheduler": args.scheduler,
            "loss": args.loss,
            "best_epoch": read_metric_value(best_metrics_path, "best_epoch"),
            "val_mean_iou": read_metric_value(best_metrics_path, "val_mean_iou"),
            "test_mean_iou": read_metric_value(metrics_path, "mean_iou"),
            "test_pixel_accuracy": read_metric_value(metrics_path, "pixel_accuracy"),
        },
    )


def print_results_summary(run_dir=None):
    print()
    print("=" * 80)
    print("DeepLabV3+ ResNet50 pipeline result paths")
    print("=" * 80)
    print(f"Train split:       {RUGD_TRAIN_SPLIT_PATH}")
    print(f"Val split:         {RUGD_VAL_SPLIT_PATH}")
    print(f"Test split:        {RUGD_TEST_SPLIT_PATH}")
    print(f"Last checkpoint:   {DEEPLABV3PLUS_RESNET50_CHECKPOINT_PATH}")
    print(f"Best checkpoint:   {DEEPLABV3PLUS_RESNET50_BEST_CHECKPOINT_PATH}")
    print(f"History:           {DEEPLABV3PLUS_RESNET50_HISTORY_PATH}")
    print(f"Step history:      {DEEPLABV3PLUS_RESNET50_STEP_HISTORY_PATH}")
    print(f"Loss history:      {DEEPLABV3PLUS_RESNET50_LOSS_HISTORY_PATH}")
    print(f"Metrics:           {DEEPLABV3PLUS_RESNET50_METRICS_PATH}")
    print(f"Best metrics:      {DEEPLABV3PLUS_RESNET50_BEST_METRICS_PATH}")
    print(f"Training curves:   {DEEPLABV3PLUS_RESNET50_TRAINING_CURVES_PATH}")
    print(f"Step loss curve:   {DEEPLABV3PLUS_RESNET50_STEP_LOSS_CURVE_PATH}")
    print(f"Predictions grid:  {DEEPLABV3PLUS_RESNET50_PREDICTIONS_GRID_PATH}")
    if run_dir is not None:
        print(f"Run dir:           {run_dir}")
        print(f"Summary CSV:       {EXPERIMENTS_SUMMARY_PATH}")


def run_pipeline(args):
    steps = collect_steps(args)

    if not steps:
        print("No steps selected. Use --main, --full, --train, --evaluate, or --help.")
        return 0

    run_dir = None
    if any(step in RUN_AWARE_STEPS for step in steps):
        run_dir = prepare_run_dir(
            DEEPLABV3PLUS_RESNET50_OUTPUT_DIR,
            run_id=args.run_id,
            run_dir=args.run_dir,
        )

    train_args = [
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--optimizer",
        args.optimizer,
        "--weight-decay",
        str(args.weight_decay),
        "--scheduler",
        args.scheduler,
        "--loss",
        args.loss,
    ]

    print("Selected DeepLabV3+ ResNet50 pipeline steps:")
    for index, step in enumerate(steps, start=1):
        print(f"{index}. {step}")
    if run_dir is not None:
        print(f"Run dir: {run_dir}")

    for step in steps:
        run_script(step, run_dir=run_dir, train_args=train_args)

    append_run_summary(run_dir, args)
    print_results_summary(run_dir)
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RUGD DeepLabV3+ ResNet50 pipeline steps.",
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Run preprocessing, split creation, training, evaluation, and visualization.",
    )
    parser.add_argument(
        "--main",
        action="store_true",
        help="Run split creation, training, evaluation, and visualization.",
    )
    parser.add_argument(
        "--checks",
        action="store_true",
        help="Run forward, loss, and one-step training checks.",
    )

    parser.add_argument("--prepare-full", action="store_true")
    parser.add_argument("--create-splits", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=DEEPLABV3PLUS_RESNET50_NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEEPLABV3PLUS_RESNET50_LEARNING_RATE)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default=DEEPLABV3PLUS_RESNET50_OPTIMIZER)
    parser.add_argument("--weight-decay", type=float, default=DEEPLABV3PLUS_RESNET50_WEIGHT_DECAY)
    parser.add_argument("--scheduler", choices=["none", "plateau"], default=DEEPLABV3PLUS_RESNET50_SCHEDULER)
    parser.add_argument("--loss", choices=["ce", "weighted_ce"], default=DEEPLABV3PLUS_RESNET50_LOSS)

    parser.add_argument("--check-dataloader", action="store_true")
    parser.add_argument("--check-forward", action="store_true")
    parser.add_argument("--check-loss", action="store_true")
    parser.add_argument("--check-train-step", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    raise SystemExit(run_pipeline(args))


if __name__ == "__main__":
    main()
