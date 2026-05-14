from __future__ import annotations

from datetime import datetime
from pathlib import Path
import csv
import shutil


def create_run_id():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def prepare_run_dir(model_output_dir, run_id=None, run_dir=None):
    if run_dir is not None:
        path = Path(run_dir)
    else:
        path = Path(model_output_dir) / "runs" / (run_id or create_run_id())

    path.mkdir(parents=True, exist_ok=True)
    return path


def get_run_artifact_path(default_path, run_dir):
    if run_dir is None:
        return Path(default_path)

    return Path(run_dir) / Path(default_path).name


def copy_existing_artifacts(paths, run_dir):
    if run_dir is None:
        return

    Path(run_dir).mkdir(parents=True, exist_ok=True)

    for path in paths:
        source_path = Path(path)
        if source_path.exists():
            shutil.copy2(source_path, Path(run_dir) / source_path.name)


def save_config_snapshot(run_dir, values):
    if run_dir is None:
        return

    snapshot_path = Path(run_dir) / "config_snapshot.txt"

    with snapshot_path.open("w", encoding="utf-8") as file:
        for key, value in values.items():
            file.write(f"{key}: {value}\n")


def read_metric_value(path, metric_name):
    path = Path(path)
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 2 and row[0] == metric_name:
                return row[1]

    return None


def append_experiment_summary(summary_path, row):
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = summary_path.exists()

    fieldnames = [
        "run_id",
        "model",
        "run_dir",
        "epochs",
        "learning_rate",
        "optimizer",
        "weight_decay",
        "scheduler",
        "loss",
        "best_epoch",
        "val_mean_iou",
        "test_mean_iou",
        "test_pixel_accuracy",
    ]

    existing_rows = []
    if file_exists:
        with summary_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != fieldnames:
                existing_rows = list(reader)
                file_exists = False

    if existing_rows:
        with summary_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for existing_row in existing_rows:
                writer.writerow(existing_row)
        file_exists = True

    with summary_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
