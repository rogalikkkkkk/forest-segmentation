from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from config import (
    RUGD_NUM_CLASSES,
    SEGFORMER_B0_BEST_CHECKPOINT_PATH,
    SEGFORMER_B0_ENCODER_WEIGHTS,
    SEGFORMER_B0_PREDICTIONS_GRID_PATH,
    SEGFORMER_B0_VISUALIZATION_INDICES,
)
from models.segformer_b0 import create_segformer_b0
from visualization_runner import parse_visualization_args, visualize_predictions


def create_model():
    return create_segformer_b0(
        num_classes=RUGD_NUM_CLASSES,
        encoder_weights=SEGFORMER_B0_ENCODER_WEIGHTS,
    )


def main():
    args = parse_visualization_args("Visualize SegFormer-B0 predictions.")
    visualize_predictions(
        display_name="SegFormer-B0",
        create_model=create_model,
        default_best_checkpoint_path=SEGFORMER_B0_BEST_CHECKPOINT_PATH,
        default_predictions_grid_path=SEGFORMER_B0_PREDICTIONS_GRID_PATH,
        visualization_indices=SEGFORMER_B0_VISUALIZATION_INDICES,
        train_script_name="train_segformer_b0.py",
        run_dir=args.run_dir,
    )


if __name__ == "__main__":
    main()
