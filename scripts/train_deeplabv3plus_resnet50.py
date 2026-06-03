from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from model_factory import create_model
from model_specs import DEEPLABV3PLUS_RESNET50_SPEC
from training_runner import run_training_cli


if __name__ == "__main__":
    run_training_cli(
        DEEPLABV3PLUS_RESNET50_SPEC,
        lambda: create_model(DEEPLABV3PLUS_RESNET50_SPEC),
    )
