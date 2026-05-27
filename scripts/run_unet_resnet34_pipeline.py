from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from model_specs import UNET_RESNET34_SPEC
from pipeline_runner import run_pipeline_cli


if __name__ == "__main__":
    run_pipeline_cli(UNET_RESNET34_SPEC)
