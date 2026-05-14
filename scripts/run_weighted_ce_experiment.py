from pathlib import Path
from datetime import datetime
import argparse
import csv
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from experiment_utils import read_metric_value


MODEL_PIPELINES = {
    "unet_resnet34": "run_unet_resnet34_pipeline.py",
    "deeplabv3plus_resnet50": "run_deeplabv3plus_resnet50_pipeline.py",
    "segformer_b0": "run_segformer_b0_pipeline.py",
}


def run_command(command):
    print()
    print("=" * 80)
    print("Command:")
    print(" ".join(str(part) for part in command))
    print("=" * 80)

    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def run_model(model_name, run_id, epochs, skip_visualize):
    pipeline_path = SCRIPTS_DIR / MODEL_PIPELINES[model_name]
    command = [
        sys.executable,
        str(pipeline_path),
        "--train",
        "--evaluate",
        "--run-id",
        run_id,
        "--epochs",
        str(epochs),
        "--loss",
        "weighted_ce",
    ]

    if not skip_visualize:
        command.append("--visualize")

    run_command(command)


def get_run_dir(model_name, run_id):
    return PROJECT_ROOT / "outputs" / model_name / "runs" / run_id


def collect_result(model_name, run_id):
    run_dir = get_run_dir(model_name, run_id)
    best_metrics_path = run_dir / "best_metrics.txt"
    metrics_path = run_dir / "metrics.txt"

    return {
        "model": model_name,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "loss": "weighted_ce",
        "best_epoch": read_metric_value(best_metrics_path, "best_epoch"),
        "val_mean_iou": read_metric_value(best_metrics_path, "val_mean_iou"),
        "val_pixel_accuracy": read_metric_value(
            best_metrics_path,
            "val_pixel_accuracy",
        ),
        "test_mean_iou": read_metric_value(metrics_path, "mean_iou"),
        "test_pixel_accuracy": read_metric_value(metrics_path, "pixel_accuracy"),
    }


def save_results(results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.csv"

    fieldnames = [
        "model",
        "run_id",
        "run_dir",
        "loss",
        "best_epoch",
        "val_mean_iou",
        "val_pixel_accuracy",
        "test_mean_iou",
        "test_pixel_accuracy",
    ]

    with results_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    return results_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run weighted CrossEntropyLoss experiment for all models.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--prefix", default="weighted_ce_ep10")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_PIPELINES.keys()),
        default=list(MODEL_PIPELINES.keys()),
    )
    parser.add_argument("--skip-visualize", action="store_true")
    parser.add_argument("--skip-compute-weights", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_id = f"{args.prefix}_{timestamp}"
    output_dir = PROJECT_ROOT / "outputs" / "loss_experiments" / experiment_id

    print("Weighted CE experiment")
    print("=" * 80)
    print(f"Models: {', '.join(args.models)}")
    print(f"Epochs: {args.epochs}")
    print(f"Experiment dir: {output_dir}")

    if not args.skip_compute_weights:
        run_command([sys.executable, str(SCRIPTS_DIR / "compute_rugd_class_weights.py")])

    results = []
    for model_name in args.models:
        run_id = f"{experiment_id}_{model_name}"
        run_model(
            model_name=model_name,
            run_id=run_id,
            epochs=args.epochs,
            skip_visualize=args.skip_visualize,
        )
        result = collect_result(model_name, run_id)
        results.append(result)
        results_path = save_results(results, output_dir)

        print()
        print(f"Saved intermediate results to: {results_path}")
        print(
            f"{model_name}: val mIoU={result['val_mean_iou']}, "
            f"test mIoU={result['test_mean_iou']}"
        )

    results_path = save_results(results, output_dir)
    print()
    print("=" * 80)
    print("Weighted CE experiment finished.")
    print(f"Results CSV: {results_path}")


if __name__ == "__main__":
    main()
