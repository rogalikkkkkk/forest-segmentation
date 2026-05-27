from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from config import RUGD_NUM_CLASSES
from model_specs import DEEPLABV3PLUS_RESNET50_SPEC
from models.deeplabv3plus_resnet50 import create_deeplabv3plus_resnet50
from training_runner import run_training_cli


def create_model():
    return create_deeplabv3plus_resnet50(
        num_classes=RUGD_NUM_CLASSES,
        encoder_weights=DEEPLABV3PLUS_RESNET50_SPEC.encoder_weights,
    )


if __name__ == "__main__":
    run_training_cli(DEEPLABV3PLUS_RESNET50_SPEC, create_model)
