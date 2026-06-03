from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from model_specs import DEEPLABV3PLUS_RESNET50_SPEC
from pipeline_runner import run_pipeline_cli


if __name__ == "__main__":
    run_pipeline_cli(DEEPLABV3PLUS_RESNET50_SPEC)
