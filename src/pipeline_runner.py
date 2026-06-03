from pathlib import Path
import argparse
import subprocess
import sys

from config import (
    EXPERIMENTS_SUMMARY_PATH,
    RUGD_TEST_SPLIT_PATH,
    RUGD_TRAIN_SPLIT_PATH,
    RUGD_VAL_SPLIT_PATH,
)
from experiment_utils import (
    append_experiment_summary,
    get_run_artifact_path,
    prepare_run_dir,
    read_metric_value,
)
from model_specs import ModelSpec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

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

RUN_AWARE_STEPS = {"train", "evaluate", "visualize"}


def get_pipeline_steps(model_spec: ModelSpec):
    return {
        "prepare_full": "prepare_rugd_full.py",
        "create_splits": "create_rugd_splits.py",
        "train": model_spec.train_script_name,
        "evaluate": model_spec.evaluate_script_name,
        "visualize": model_spec.visualize_script_name,
    }


def run_script(model_spec: ModelSpec, step_name, run_dir=None, train_args=None):
    script_name = get_pipeline_steps(model_spec)[step_name]
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

    ordered_unique_steps = []
    seen = set()

    for step in steps:
        if step not in seen:
            ordered_unique_steps.append(step)
            seen.add(step)

    return ordered_unique_steps


def append_run_summary(model_spec: ModelSpec, run_dir, args):
    if run_dir is None:
        return

    best_metrics_path = run_dir / model_spec.best_metrics_path.name
    metrics_path = run_dir / model_spec.metrics_path.name

    append_experiment_summary(
        EXPERIMENTS_SUMMARY_PATH,
        {
            "run_id": run_dir.name,
            "model": model_spec.name,
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


def print_results_summary(model_spec: ModelSpec, run_dir=None):
    checkpoint_path = get_run_artifact_path(model_spec.checkpoint_path, run_dir)
    best_checkpoint_path = get_run_artifact_path(
        model_spec.best_checkpoint_path,
        run_dir,
    )
    history_path = get_run_artifact_path(model_spec.history_path, run_dir)
    step_history_path = get_run_artifact_path(model_spec.step_history_path, run_dir)
    metrics_path = get_run_artifact_path(model_spec.metrics_path, run_dir)
    best_metrics_path = get_run_artifact_path(model_spec.best_metrics_path, run_dir)
    training_curves_path = get_run_artifact_path(
        model_spec.training_curves_path,
        run_dir,
    )
    step_loss_curve_path = get_run_artifact_path(
        model_spec.step_loss_curve_path,
        run_dir,
    )
    predictions_grid_path = get_run_artifact_path(
        model_spec.predictions_grid_path,
        run_dir,
    )

    print()
    print("=" * 80)
    print(f"{model_spec.display_name} pipeline result paths")
    print("=" * 80)
    print(f"Train split:       {RUGD_TRAIN_SPLIT_PATH}")
    print(f"Val split:         {RUGD_VAL_SPLIT_PATH}")
    print(f"Test split:        {RUGD_TEST_SPLIT_PATH}")
    print(f"Last checkpoint:   {checkpoint_path}")
    print(f"Best checkpoint:   {best_checkpoint_path}")
    print(f"History:           {history_path}")
    print(f"Step history:      {step_history_path}")
    print(f"Metrics:           {metrics_path}")
    print(f"Best metrics:      {best_metrics_path}")
    print(f"Training curves:   {training_curves_path}")
    print(f"Step loss curve:   {step_loss_curve_path}")
    print(f"Predictions grid:  {predictions_grid_path}")
    if run_dir is not None:
        print(f"Run dir:           {run_dir}")
        print(f"Summary CSV:       {EXPERIMENTS_SUMMARY_PATH}")


def resolve_run_dir(model_spec: ModelSpec, steps, args):
    if "train" in steps:
        return prepare_run_dir(
            model_spec.output_dir,
            run_id=args.run_id,
            run_dir=args.run_dir,
        )

    if not any(step in RUN_AWARE_STEPS for step in steps):
        return None

    if args.run_dir is not None:
        return Path(args.run_dir)

    if args.run_id is not None:
        return model_spec.output_dir / "runs" / args.run_id

    return None


def run_pipeline(model_spec: ModelSpec, args):
    steps = collect_steps(args)

    if not steps:
        print("No steps selected. Use --main, --full, --train, --evaluate, or --help.")
        return 0

    run_dir = resolve_run_dir(model_spec, steps, args)

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
        "--prediction-interval",
        str(args.prediction_interval),
        "--prediction-epochs",
        *[str(epoch) for epoch in args.prediction_epochs],
        "--prediction-sample-count",
        str(args.prediction_sample_count),
        "--prediction-seed",
        str(args.prediction_seed),
    ]
    if args.disable_epoch_predictions:
        train_args.append("--disable-epoch-predictions")

    print(f"Selected {model_spec.display_name} pipeline steps:")
    for index, step in enumerate(steps, start=1):
        print(f"{index}. {step}")
    if run_dir is not None:
        print(f"Run dir: {run_dir}")

    for step in steps:
        run_script(model_spec, step, run_dir=run_dir, train_args=train_args)

    append_run_summary(model_spec, run_dir, args)
    print_results_summary(model_spec, run_dir)
    return 0


def parse_pipeline_args(model_spec: ModelSpec):
    parser = argparse.ArgumentParser(
        description=f"Run RUGD {model_spec.display_name} pipeline steps.",
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
    parser.add_argument("--prepare-full", action="store_true")
    parser.add_argument("--create-splits", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=model_spec.num_epochs)
    parser.add_argument("--lr", type=float, default=model_spec.learning_rate)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default=model_spec.optimizer)
    parser.add_argument("--weight-decay", type=float, default=model_spec.weight_decay)
    parser.add_argument("--scheduler", choices=["none", "plateau"], default=model_spec.scheduler)
    parser.add_argument("--loss", choices=["ce", "weighted_ce"], default=model_spec.loss)
    parser.add_argument("--prediction-interval", type=int, default=5)
    parser.add_argument("--prediction-epochs", nargs="+", type=int, default=[1, 5, 10, 15, 20, 25, 30])
    parser.add_argument("--prediction-sample-count", type=int, default=5)
    parser.add_argument("--prediction-seed", type=int, default=42)
    parser.add_argument("--disable-epoch-predictions", action="store_true")

    return parser.parse_args()


def run_pipeline_cli(model_spec: ModelSpec):
    args = parse_pipeline_args(model_spec)
    raise SystemExit(run_pipeline(model_spec, args))
