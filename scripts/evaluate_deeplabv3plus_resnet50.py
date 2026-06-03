from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from evaluation_runner import evaluate_model, parse_evaluation_args
from model_factory import create_model
from model_specs import DEEPLABV3PLUS_RESNET50_SPEC


def main():
    args = parse_evaluation_args("Evaluate DeepLabV3+ ResNet50 on RUGD test split.")
    evaluate_model(
        display_name=DEEPLABV3PLUS_RESNET50_SPEC.display_name,
        create_model=lambda: create_model(DEEPLABV3PLUS_RESNET50_SPEC),
        default_best_checkpoint_path=DEEPLABV3PLUS_RESNET50_SPEC.best_checkpoint_path,
        default_metrics_path=DEEPLABV3PLUS_RESNET50_SPEC.metrics_path,
        train_script_name=DEEPLABV3PLUS_RESNET50_SPEC.train_script_name,
        run_dir=args.run_dir,
    )


if __name__ == "__main__":
    main()
