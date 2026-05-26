from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from config import (
    RUGD_NUM_CLASSES,
    UNET_RESNET34_BEST_CHECKPOINT_PATH,
    UNET_RESNET34_ENCODER_WEIGHTS,
    UNET_RESNET34_METRICS_PATH,
)
from evaluation_runner import evaluate_model, parse_evaluation_args
from models.unet_resnet34 import UNetResNet34


def create_model():
    return UNetResNet34(
        num_classes=RUGD_NUM_CLASSES,
        encoder_weights=UNET_RESNET34_ENCODER_WEIGHTS,
    )


def main():
    args = parse_evaluation_args("Evaluate U-Net ResNet34 on RUGD test split.")
    evaluate_model(
        display_name="U-Net ResNet34",
        create_model=create_model,
        default_best_checkpoint_path=UNET_RESNET34_BEST_CHECKPOINT_PATH,
        default_metrics_path=UNET_RESNET34_METRICS_PATH,
        train_script_name="train_unet_resnet34.py",
        run_dir=args.run_dir,
    )


if __name__ == "__main__":
    main()
