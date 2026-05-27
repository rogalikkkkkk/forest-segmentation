from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from config import RUGD_NUM_CLASSES
from model_specs import UNET_RESNET34_SPEC
from models.unet_resnet34 import UNetResNet34
from training_runner import run_training_cli


def create_model():
    return UNetResNet34(
        num_classes=RUGD_NUM_CLASSES,
        encoder_weights=UNET_RESNET34_SPEC.encoder_weights,
    )


if __name__ == "__main__":
    run_training_cli(UNET_RESNET34_SPEC, create_model)
