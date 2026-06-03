from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from model_factory import create_model
from model_specs import SEGFORMER_B0_SPEC
from training_runner import run_training_cli


if __name__ == "__main__":
    run_training_cli(SEGFORMER_B0_SPEC, lambda: create_model(SEGFORMER_B0_SPEC))
