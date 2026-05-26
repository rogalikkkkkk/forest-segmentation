from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from config import (
    DEEPLABV3PLUS_RESNET50_BEST_CHECKPOINT_PATH,
    DEEPLABV3PLUS_RESNET50_ENCODER_WEIGHTS,
    DEEPLABV3PLUS_RESNET50_PREDICTIONS_GRID_PATH,
    DEEPLABV3PLUS_RESNET50_VISUALIZATION_INDICES,
    RUGD_NUM_CLASSES,
)
from models.deeplabv3plus_resnet50 import create_deeplabv3plus_resnet50
from visualization_runner import parse_visualization_args, visualize_predictions


def create_model():
    return create_deeplabv3plus_resnet50(
        num_classes=RUGD_NUM_CLASSES,
        encoder_weights=DEEPLABV3PLUS_RESNET50_ENCODER_WEIGHTS,
    )


def main():
    args = parse_visualization_args("Visualize DeepLabV3+ ResNet50 predictions.")
    visualize_predictions(
        display_name="DeepLabV3+ ResNet50",
        create_model=create_model,
        default_best_checkpoint_path=DEEPLABV3PLUS_RESNET50_BEST_CHECKPOINT_PATH,
        default_predictions_grid_path=DEEPLABV3PLUS_RESNET50_PREDICTIONS_GRID_PATH,
        visualization_indices=DEEPLABV3PLUS_RESNET50_VISUALIZATION_INDICES,
        train_script_name="train_deeplabv3plus_resnet50.py",
        run_dir=args.run_dir,
    )


if __name__ == "__main__":
    main()
