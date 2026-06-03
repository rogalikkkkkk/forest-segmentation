from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from evaluation_runner import evaluate_model, parse_evaluation_args
from model_factory import create_model
from model_specs import UNET_RESNET34_SPEC


def main():
    args = parse_evaluation_args("Evaluate U-Net ResNet34 on RUGD test split.")
    evaluate_model(
        display_name=UNET_RESNET34_SPEC.display_name,
        create_model=lambda: create_model(UNET_RESNET34_SPEC),
        default_best_checkpoint_path=UNET_RESNET34_SPEC.best_checkpoint_path,
        default_metrics_path=UNET_RESNET34_SPEC.metrics_path,
        train_script_name=UNET_RESNET34_SPEC.train_script_name,
        run_dir=args.run_dir,
    )


if __name__ == "__main__":
    main()
