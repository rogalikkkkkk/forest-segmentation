from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from config import RUGD_NUM_CLASSES
from model_specs import SEGFORMER_B0_SPEC
from models.segformer_b0 import create_segformer_b0
from training_runner import run_training_cli


def create_model():
    return create_segformer_b0(
        num_classes=RUGD_NUM_CLASSES,
        encoder_weights=SEGFORMER_B0_SPEC.encoder_weights,
    )


if __name__ == "__main__":
    run_training_cli(SEGFORMER_B0_SPEC, create_model)
