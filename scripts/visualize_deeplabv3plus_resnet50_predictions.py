from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from config import (
    DEEPLABV3PLUS_RESNET50_VISUALIZATION_INDICES,
)
from model_factory import create_model
from model_specs import DEEPLABV3PLUS_RESNET50_SPEC
from visualization_runner import parse_visualization_args, visualize_predictions


def main():
    args = parse_visualization_args("Visualize DeepLabV3+ ResNet50 predictions.")
    visualize_predictions(
        display_name=DEEPLABV3PLUS_RESNET50_SPEC.display_name,
        create_model=lambda: create_model(DEEPLABV3PLUS_RESNET50_SPEC),
        default_best_checkpoint_path=DEEPLABV3PLUS_RESNET50_SPEC.best_checkpoint_path,
        default_predictions_grid_path=DEEPLABV3PLUS_RESNET50_SPEC.predictions_grid_path,
        visualization_indices=DEEPLABV3PLUS_RESNET50_VISUALIZATION_INDICES,
        train_script_name=DEEPLABV3PLUS_RESNET50_SPEC.train_script_name,
        run_dir=args.run_dir,
    )


if __name__ == "__main__":
    main()
