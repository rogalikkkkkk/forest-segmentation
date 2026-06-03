from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from model_factory import create_model
from model_specs import SEGFORMER_B0_SPEC
from visualization_runner import parse_visualization_args, visualize_predictions


def main():
    args = parse_visualization_args("Visualize SegFormer-B0 predictions.")
    visualize_predictions(
        display_name=SEGFORMER_B0_SPEC.display_name,
        create_model=lambda: create_model(SEGFORMER_B0_SPEC),
        default_best_checkpoint_path=SEGFORMER_B0_SPEC.best_checkpoint_path,
        default_predictions_grid_path=SEGFORMER_B0_SPEC.predictions_grid_path,
        visualization_indices=SEGFORMER_B0_SPEC.visualization_indices,
        train_script_name=SEGFORMER_B0_SPEC.train_script_name,
        run_dir=args.run_dir,
    )


if __name__ == "__main__":
    main()
