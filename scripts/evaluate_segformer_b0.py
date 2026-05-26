from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from config import (
    RUGD_NUM_CLASSES,
    SEGFORMER_B0_BEST_CHECKPOINT_PATH,
    SEGFORMER_B0_ENCODER_WEIGHTS,
    SEGFORMER_B0_METRICS_PATH,
)
from evaluation_runner import evaluate_model, parse_evaluation_args
from models.segformer_b0 import create_segformer_b0


def create_model():
    return create_segformer_b0(
        num_classes=RUGD_NUM_CLASSES,
        encoder_weights=SEGFORMER_B0_ENCODER_WEIGHTS,
    )


def main():
    args = parse_evaluation_args("Evaluate SegFormer-B0 on RUGD test split.")
    evaluate_model(
        display_name="SegFormer-B0",
        create_model=create_model,
        default_best_checkpoint_path=SEGFORMER_B0_BEST_CHECKPOINT_PATH,
        default_metrics_path=SEGFORMER_B0_METRICS_PATH,
        train_script_name="train_segformer_b0.py",
        run_dir=args.run_dir,
    )


if __name__ == "__main__":
    main()
