from pathlib import Path
import argparse
import csv
import subprocess
import sys
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from experiment_utils import read_metric_value


MODELS = {
    "unet": {
        "name": "unet_resnet34",
        "pipeline": "run_unet_resnet34_pipeline.py",
        "output_dir": PROJECT_ROOT / "outputs" / "unet_resnet34",
    },
    "deeplab": {
        "name": "deeplabv3plus_resnet50",
        "pipeline": "run_deeplabv3plus_resnet50_pipeline.py",
        "output_dir": PROJECT_ROOT / "outputs" / "deeplabv3plus_resnet50",
    },
    "segformer": {
        "name": "segformer_b0",
        "pipeline": "run_segformer_b0_pipeline.py",
        "output_dir": PROJECT_ROOT / "outputs" / "segformer_b0",
    },
}


BASE_PARAMS = {
    "lr": 1e-4,
    "optimizer": "adam",
    "weight_decay": 0.0,
    "scheduler": "none",
}


EXPERIMENTS = {
    "lr": [
        {"label": "lr1e-4", "lr": 1e-4},
        {"label": "lr3e-5", "lr": 3e-5},
        {"label": "lr1e-5", "lr": 1e-5},
    ],
    "optimizer": [
        {"label": "adam", "optimizer": "adam"},
        {"label": "adamw", "optimizer": "adamw"},
    ],
    "weight_decay": [
        {"label": "wd0", "optimizer": "adamw", "weight_decay": 0.0},
        {"label": "wd1e-5", "optimizer": "adamw", "weight_decay": 1e-5},
        {"label": "wd1e-4", "optimizer": "adamw", "weight_decay": 1e-4},
    ],
    "scheduler": [
        {"label": "sched_none", "scheduler": "none"},
        {"label": "sched_plateau", "scheduler": "plateau"},
    ],
}


def build_experiment_plan(parameter):
    if parameter == "all":
        plan = []
        for parameter_name, variants in EXPERIMENTS.items():
            for variant in variants:
                plan.append((parameter_name, variant))
        return plan

    return [(parameter, variant) for variant in EXPERIMENTS[parameter]]


def select_models(model_arg):
    if model_arg == "all":
        return list(MODELS.items())

    return [(model_arg, MODELS[model_arg])]


def merge_params(variant):
    params = dict(BASE_PARAMS)
    params.update(variant)
    return params


def format_float_for_run_id(value):
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def build_run_id(prefix, model_name, parameter_name, variant_label, epochs, params):
    lr_label = format_float_for_run_id(params["lr"])
    weight_decay_label = format_float_for_run_id(params["weight_decay"])
    return (
        f"{prefix}_{model_name}_{parameter_name}_{variant_label}"
        f"_ep{epochs}_lr{lr_label}_{params['optimizer']}"
        f"_wd{weight_decay_label}_{params['scheduler']}"
    )


def build_command(model_info, run_id, epochs, params, skip_visualize):
    command = [
        sys.executable,
        str(SCRIPTS_DIR / model_info["pipeline"]),
        "--train",
        "--evaluate",
        "--run-id",
        run_id,
        "--epochs",
        str(epochs),
        "--lr",
        str(params["lr"]),
        "--optimizer",
        params["optimizer"],
        "--weight-decay",
        str(params["weight_decay"]),
        "--scheduler",
        params["scheduler"],
    ]

    if not skip_visualize:
        command.append("--visualize")

    return command


def read_run_results(run_dir):
    best_metrics_path = run_dir / "best_metrics.txt"
    metrics_path = run_dir / "metrics.txt"

    return {
        "best_epoch": read_metric_value(best_metrics_path, "best_epoch"),
        "val_mean_iou": read_metric_value(best_metrics_path, "val_mean_iou"),
        "val_loss": read_metric_value(best_metrics_path, "val_loss"),
        "test_mean_iou": read_metric_value(metrics_path, "mean_iou"),
        "test_pixel_accuracy": read_metric_value(metrics_path, "pixel_accuracy"),
    }


def write_plan_csv(path, rows):
    fieldnames = [
        "status",
        "parameter",
        "variant",
        "model",
        "epochs",
        "learning_rate",
        "optimizer",
        "weight_decay",
        "scheduler",
        "run_id",
        "run_dir",
        "best_epoch",
        "val_mean_iou",
        "val_loss",
        "test_mean_iou",
        "test_pixel_accuracy",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run 10-epoch hyperparameter experiments for U-Net ResNet34, "
            "DeepLabV3+ ResNet50, and SegFormer-B0."
        ),
    )
    parser.add_argument(
        "--parameter",
        choices=["lr", "optimizer", "weight_decay", "scheduler", "all"],
        required=True,
        help="Hyperparameter to study. Use 'all' only if you are ready for many runs.",
    )
    parser.add_argument(
        "--models",
        choices=["all", "unet", "deeplab", "segformer"],
        default="all",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--prefix",
        default=None,
        help="Prefix for run IDs. Defaults to hp_<parameter>_<timestamp>.",
    )
    parser.add_argument(
        "--skip-visualize",
        action="store_true",
        help="Train and evaluate only, without prediction grids.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the whole experiment if one run fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without running training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    prefix = args.prefix or f"hp_{args.parameter}_{timestamp}"

    experiment_dir = PROJECT_ROOT / "outputs" / "hyperparameter_experiments" / prefix
    experiment_dir.mkdir(parents=True, exist_ok=True)
    plan_csv_path = experiment_dir / "results.csv"
    log_path = experiment_dir / "experiment_log.txt"

    planned_rows = []
    plan = build_experiment_plan(args.parameter)
    selected_models = select_models(args.models)

    print("Hyperparameter experiment")
    print("=" * 80)
    print(f"Parameter: {args.parameter}")
    print(f"Models: {args.models}")
    print(f"Epochs: {args.epochs}")
    print(f"Experiment dir: {experiment_dir}")
    print(f"Results CSV: {plan_csv_path}")
    print()

    with log_path.open("w", encoding="utf-8") as log_file:
        for parameter_name, variant in plan:
            params = merge_params(variant)

            for model_key, model_info in selected_models:
                run_id = build_run_id(
                    prefix=prefix,
                    model_name=model_info["name"],
                    parameter_name=parameter_name,
                    variant_label=variant["label"],
                    epochs=args.epochs,
                    params=params,
                )
                run_dir = model_info["output_dir"] / "runs" / run_id
                command = build_command(
                    model_info=model_info,
                    run_id=run_id,
                    epochs=args.epochs,
                    params=params,
                    skip_visualize=args.skip_visualize,
                )

                row = {
                    "status": "planned",
                    "parameter": parameter_name,
                    "variant": variant["label"],
                    "model": model_info["name"],
                    "epochs": args.epochs,
                    "learning_rate": params["lr"],
                    "optimizer": params["optimizer"],
                    "weight_decay": params["weight_decay"],
                    "scheduler": params["scheduler"],
                    "run_id": run_id,
                    "run_dir": run_dir,
                    "best_epoch": None,
                    "val_mean_iou": None,
                    "val_loss": None,
                    "test_mean_iou": None,
                    "test_pixel_accuracy": None,
                }
                planned_rows.append(row)
                write_plan_csv(plan_csv_path, planned_rows)

                print("=" * 80)
                print(f"Run: {run_id}")
                print("Command:")
                print(" ".join(str(part) for part in command))
                print()

                log_file.write("=" * 80 + "\n")
                log_file.write(f"Run: {run_id}\n")
                log_file.write("Command:\n")
                log_file.write(" ".join(str(part) for part in command) + "\n")
                log_file.flush()

                if args.dry_run:
                    row["status"] = "dry_run"
                    write_plan_csv(plan_csv_path, planned_rows)
                    continue

                row["status"] = "running"
                write_plan_csv(plan_csv_path, planned_rows)

                try:
                    subprocess.run(
                        command,
                        cwd=PROJECT_ROOT,
                        check=True,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                    )
                except subprocess.CalledProcessError as error:
                    row["status"] = f"failed_exit_{error.returncode}"
                    write_plan_csv(plan_csv_path, planned_rows)
                    log_file.write(f"FAILED: {error}\n")
                    log_file.flush()

                    if args.stop_on_error:
                        raise

                    continue

                results = read_run_results(run_dir)
                row.update(results)
                row["status"] = "completed"
                write_plan_csv(plan_csv_path, planned_rows)

                log_file.write(f"COMPLETED: {run_id}\n")
                log_file.flush()

    print("Experiment finished.")
    print(f"Results CSV: {plan_csv_path}")
    print(f"Log file: {log_path}")


if __name__ == "__main__":
    main()
