from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from model_factory import create_model
from model_specs import UNET_RESNET34_SPEC
from visualization_runner import parse_visualization_args, visualize_predictions


def main():
    args = parse_visualization_args("Visualize U-Net ResNet34 predictions.")
    visualize_predictions(
        display_name=UNET_RESNET34_SPEC.display_name,
        create_model=lambda: create_model(UNET_RESNET34_SPEC),
        default_best_checkpoint_path=UNET_RESNET34_SPEC.best_checkpoint_path,
        default_predictions_grid_path=UNET_RESNET34_SPEC.predictions_grid_path,
        visualization_indices=UNET_RESNET34_SPEC.visualization_indices,
        train_script_name=UNET_RESNET34_SPEC.train_script_name,
        run_dir=args.run_dir,
    )


if __name__ == "__main__":
    main()
