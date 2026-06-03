from pathlib import Path
import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime

import optuna


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from experiment_utils import read_metric_value
from model_specs import MODEL_SPECS_BY_ALIAS, ModelSpec


def select_models(model_arg):
    if model_arg == "all":
        return list(MODEL_SPECS_BY_ALIAS.items())

    return [(model_arg, MODEL_SPECS_BY_ALIAS[model_arg])]


def format_float_for_run_id(value):
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def suggest_params(trial):
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
    optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw"])
    scheduler = trial.suggest_categorical("scheduler", ["none", "plateau"])

    if optimizer == "adamw":
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    else:
        weight_decay = 0.0

    return {
        "lr": learning_rate,
        "optimizer": optimizer,
        "weight_decay": weight_decay,
        "scheduler": scheduler,
    }


def build_trial_run_id(prefix, model_name, trial_number, params, epochs):
    lr_label = format_float_for_run_id(params["lr"])
    weight_decay_label = format_float_for_run_id(params["weight_decay"])

    return (
        f"{prefix}_{model_name}_trial{trial_number:03d}"
        f"_ep{epochs}_lr{lr_label}_{params['optimizer']}"
        f"_wd{weight_decay_label}_{params['scheduler']}"
    )


def build_pipeline_command(model_spec: ModelSpec, run_id, epochs, params, evaluate, visualize):
    command = [
        sys.executable,
        str(SCRIPTS_DIR / model_spec.pipeline_script_name),
        "--train",
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

    if evaluate:
        command.append("--evaluate")

    if visualize:
        command.append("--visualize")

    return command


def read_trial_metrics(run_dir):
    best_metrics_path = run_dir / "best_metrics.txt"
    metrics_path = run_dir / "metrics.txt"

    val_mean_iou = read_metric_value(best_metrics_path, "val_mean_iou")
    if val_mean_iou is None:
        raise RuntimeError(f"val_mean_iou not found in {best_metrics_path}")

    return {
        "best_epoch": read_metric_value(best_metrics_path, "best_epoch"),
        "val_loss": read_metric_value(best_metrics_path, "val_loss"),
        "val_pixel_accuracy": read_metric_value(best_metrics_path, "val_pixel_accuracy"),
        "val_mean_iou": val_mean_iou,
        "test_mean_iou": read_metric_value(metrics_path, "mean_iou"),
        "test_pixel_accuracy": read_metric_value(metrics_path, "pixel_accuracy"),
    }


def save_best_params(path, model_name, study, epochs):
    payload = {
        "model": model_name,
        "epochs": epochs,
        "best_value": study.best_value,
        "best_trial_number": study.best_trial.number,
        "best_params": study.best_params,
        "user_attrs": study.best_trial.user_attrs,
    }

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def save_study_results(path, study):
    fieldnames = [
        "number",
        "state",
        "value",
        "lr",
        "optimizer",
        "weight_decay",
        "scheduler",
        "run_id",
        "run_dir",
        "best_epoch",
        "val_loss",
        "val_pixel_accuracy",
        "test_mean_iou",
        "test_pixel_accuracy",
    ]

    rows = []
    for trial in study.trials:
        rows.append(
            {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "lr": trial.params.get("lr"),
                "optimizer": trial.params.get("optimizer"),
                "weight_decay": trial.params.get("weight_decay", 0.0),
                "scheduler": trial.params.get("scheduler"),
                "run_id": trial.user_attrs.get("run_id"),
                "run_dir": trial.user_attrs.get("run_dir"),
                "best_epoch": trial.user_attrs.get("best_epoch"),
                "val_loss": trial.user_attrs.get("val_loss"),
                "val_pixel_accuracy": trial.user_attrs.get("val_pixel_accuracy"),
                "test_mean_iou": trial.user_attrs.get("test_mean_iou"),
                "test_pixel_accuracy": trial.user_attrs.get("test_pixel_accuracy"),
            }
        )

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_final_training(args, model_spec: ModelSpec, model_dir, study):
    best_params = dict(study.best_params)
    if best_params["optimizer"] == "adam":
        best_params["weight_decay"] = 0.0

    run_id = f"{args.prefix}_{model_spec.name}_optuna_best_final_ep{args.final_epochs}"
    command = build_pipeline_command(
        model_spec=model_spec,
        run_id=run_id,
        epochs=args.final_epochs,
        params=best_params,
        evaluate=True,
        visualize=True,
    )

    final_log_path = model_dir / "final_run_log.txt"
    print(f"Running final training for {model_spec.name}: {run_id}")
    print(f"Final log: {final_log_path}")

    with final_log_path.open("w", encoding="utf-8") as log_file:
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )


def optimize_model(args, model_spec: ModelSpec, experiment_dir, storage_url):
    model_dir = experiment_dir / model_spec.name
    model_dir.mkdir(parents=True, exist_ok=True)

    study_name = f"{args.prefix}_{model_spec.name}"
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        storage=storage_url,
        load_if_exists=True,
    )

    log_path = model_dir / "optuna_trials_log.txt"
    results_path = model_dir / "study_results.csv"
    best_params_path = model_dir / "best_params.json"

    def objective(trial):
        params = suggest_params(trial)
        run_id = build_trial_run_id(
            prefix=args.prefix,
            model_name=model_spec.name,
            trial_number=trial.number,
            params=params,
            epochs=args.epochs,
        )
        run_dir = model_spec.output_dir / "runs" / run_id
        command = build_pipeline_command(
            model_spec=model_spec,
            run_id=run_id,
            epochs=args.epochs,
            params=params,
            evaluate=args.evaluate_trials,
            visualize=False,
        )

        trial.set_user_attr("run_id", run_id)
        trial.set_user_attr("run_dir", str(run_dir))

        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write("=" * 80 + "\n")
            log_file.write(f"Trial {trial.number}\n")
            log_file.write(f"Run ID: {run_id}\n")
            log_file.write(f"Params: {params}\n")
            log_file.write("Command:\n")
            log_file.write(" ".join(str(part) for part in command) + "\n")
            log_file.flush()

            subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                check=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )

        metrics = read_trial_metrics(run_dir)
        for key, value in metrics.items():
            trial.set_user_attr(key, value)

        save_study_results(results_path, study)

        return float(metrics["val_mean_iou"])

    study.optimize(
        objective,
        n_trials=args.trials,
        catch=(subprocess.CalledProcessError, RuntimeError),
    )

    save_study_results(results_path, study)
    save_best_params(best_params_path, model_spec.name, study, args.epochs)

    print()
    print("=" * 80)
    print(f"Optuna finished for {model_spec.name}")
    print(f"Best value: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")
    print(f"Study results: {results_path}")
    print(f"Best params JSON: {best_params_path}")

    if args.run_final:
        run_final_training(args, model_spec, model_dir, study)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Optuna hyperparameter optimization for segmentation models.",
    )
    parser.add_argument(
        "--model",
        choices=["all", *MODEL_SPECS_BY_ALIAS.keys()],
        default="all",
    )
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prefix",
        default=None,
        help="Prefix for study and run IDs. Defaults to optuna_<timestamp>.",
    )
    parser.add_argument(
        "--evaluate-trials",
        action="store_true",
        help="Also evaluate every trial on test split. Disabled by default to keep test split for final evaluation.",
    )
    parser.add_argument(
        "--run-final",
        action="store_true",
        help="After optimization, train final model with the best params and evaluate/visualize it.",
    )
    parser.add_argument("--final-epochs", type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.prefix = args.prefix or f"optuna_{timestamp}"

    experiment_dir = PROJECT_ROOT / "outputs" / "optuna" / args.prefix
    experiment_dir.mkdir(parents=True, exist_ok=True)
    storage_path = experiment_dir / "optuna_studies.db"
    storage_url = f"sqlite:///{storage_path.as_posix()}"

    print("Optuna experiment")
    print("=" * 80)
    print(f"Models: {args.model}")
    print(f"Trials per model: {args.trials}")
    print(f"Epochs per trial: {args.epochs}")
    print(f"Prefix: {args.prefix}")
    print(f"Experiment dir: {experiment_dir}")
    print(f"Storage: {storage_path}")
    print(f"Evaluate trials on test: {args.evaluate_trials}")
    print(f"Run final training: {args.run_final}")
    print()

    for _, model_spec in select_models(args.model):
        optimize_model(
            args=args,
            model_spec=model_spec,
            experiment_dir=experiment_dir,
            storage_url=storage_url,
        )

    print()
    print("Optuna experiment finished.")
    print(f"Experiment dir: {experiment_dir}")


if __name__ == "__main__":
    main()
