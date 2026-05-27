from dataclasses import dataclass
from pathlib import Path

from config import (
    DEEPLABV3PLUS_RESNET50_BEST_CHECKPOINT_PATH,
    DEEPLABV3PLUS_RESNET50_BEST_METRICS_PATH,
    DEEPLABV3PLUS_RESNET50_CHECKPOINT_PATH,
    DEEPLABV3PLUS_RESNET50_ENCODER_WEIGHTS,
    DEEPLABV3PLUS_RESNET50_HISTORY_PATH,
    DEEPLABV3PLUS_RESNET50_LEARNING_RATE,
    DEEPLABV3PLUS_RESNET50_LOG_EVERY_N_BATCHES,
    DEEPLABV3PLUS_RESNET50_LOSS,
    DEEPLABV3PLUS_RESNET50_METRICS_PATH,
    DEEPLABV3PLUS_RESNET50_NUM_EPOCHS,
    DEEPLABV3PLUS_RESNET50_OPTIMIZER,
    DEEPLABV3PLUS_RESNET50_OUTPUT_DIR,
    DEEPLABV3PLUS_RESNET50_PREDICTIONS_GRID_PATH,
    DEEPLABV3PLUS_RESNET50_SCHEDULER,
    DEEPLABV3PLUS_RESNET50_STEP_HISTORY_PATH,
    DEEPLABV3PLUS_RESNET50_STEP_LOSS_CURVE_PATH,
    DEEPLABV3PLUS_RESNET50_TRAINING_CURVES_PATH,
    DEEPLABV3PLUS_RESNET50_WEIGHT_DECAY,
    SEGFORMER_B0_BEST_CHECKPOINT_PATH,
    SEGFORMER_B0_BEST_METRICS_PATH,
    SEGFORMER_B0_CHECKPOINT_PATH,
    SEGFORMER_B0_ENCODER_WEIGHTS,
    SEGFORMER_B0_HISTORY_PATH,
    SEGFORMER_B0_LEARNING_RATE,
    SEGFORMER_B0_LOG_EVERY_N_BATCHES,
    SEGFORMER_B0_LOSS,
    SEGFORMER_B0_METRICS_PATH,
    SEGFORMER_B0_NUM_EPOCHS,
    SEGFORMER_B0_OPTIMIZER,
    SEGFORMER_B0_OUTPUT_DIR,
    SEGFORMER_B0_PREDICTIONS_GRID_PATH,
    SEGFORMER_B0_SCHEDULER,
    SEGFORMER_B0_STEP_HISTORY_PATH,
    SEGFORMER_B0_STEP_LOSS_CURVE_PATH,
    SEGFORMER_B0_TRAINING_CURVES_PATH,
    SEGFORMER_B0_WEIGHT_DECAY,
    UNET_RESNET34_BEST_CHECKPOINT_PATH,
    UNET_RESNET34_BEST_METRICS_PATH,
    UNET_RESNET34_CHECKPOINT_PATH,
    UNET_RESNET34_ENCODER_WEIGHTS,
    UNET_RESNET34_HISTORY_PATH,
    UNET_RESNET34_LEARNING_RATE,
    UNET_RESNET34_LOG_EVERY_N_BATCHES,
    UNET_RESNET34_LOSS,
    UNET_RESNET34_METRICS_PATH,
    UNET_RESNET34_NUM_EPOCHS,
    UNET_RESNET34_OPTIMIZER,
    UNET_RESNET34_OUTPUT_DIR,
    UNET_RESNET34_PREDICTIONS_GRID_PATH,
    UNET_RESNET34_SCHEDULER,
    UNET_RESNET34_STEP_HISTORY_PATH,
    UNET_RESNET34_STEP_LOSS_CURVE_PATH,
    UNET_RESNET34_TRAINING_CURVES_PATH,
    UNET_RESNET34_WEIGHT_DECAY,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    pipeline_script_name: str
    train_script_name: str
    evaluate_script_name: str
    visualize_script_name: str
    output_dir: Path
    checkpoint_path: Path
    best_checkpoint_path: Path
    history_path: Path
    step_history_path: Path
    metrics_path: Path
    best_metrics_path: Path
    training_curves_path: Path
    step_loss_curve_path: Path
    predictions_grid_path: Path
    encoder_weights: str
    log_every_n_batches: int
    num_epochs: int
    learning_rate: float
    optimizer: str
    weight_decay: float
    scheduler: str
    loss: str


UNET_RESNET34_SPEC = ModelSpec(
    name="unet_resnet34",
    display_name="U-Net ResNet34",
    pipeline_script_name="run_unet_resnet34_pipeline.py",
    train_script_name="train_unet_resnet34.py",
    evaluate_script_name="evaluate_unet_resnet34.py",
    visualize_script_name="visualize_unet_resnet34_predictions.py",
    output_dir=UNET_RESNET34_OUTPUT_DIR,
    checkpoint_path=UNET_RESNET34_CHECKPOINT_PATH,
    best_checkpoint_path=UNET_RESNET34_BEST_CHECKPOINT_PATH,
    history_path=UNET_RESNET34_HISTORY_PATH,
    step_history_path=UNET_RESNET34_STEP_HISTORY_PATH,
    metrics_path=UNET_RESNET34_METRICS_PATH,
    best_metrics_path=UNET_RESNET34_BEST_METRICS_PATH,
    training_curves_path=UNET_RESNET34_TRAINING_CURVES_PATH,
    step_loss_curve_path=UNET_RESNET34_STEP_LOSS_CURVE_PATH,
    predictions_grid_path=UNET_RESNET34_PREDICTIONS_GRID_PATH,
    encoder_weights=UNET_RESNET34_ENCODER_WEIGHTS,
    log_every_n_batches=UNET_RESNET34_LOG_EVERY_N_BATCHES,
    num_epochs=UNET_RESNET34_NUM_EPOCHS,
    learning_rate=UNET_RESNET34_LEARNING_RATE,
    optimizer=UNET_RESNET34_OPTIMIZER,
    weight_decay=UNET_RESNET34_WEIGHT_DECAY,
    scheduler=UNET_RESNET34_SCHEDULER,
    loss=UNET_RESNET34_LOSS,
)


SEGFORMER_B0_SPEC = ModelSpec(
    name="segformer_b0",
    display_name="SegFormer-B0",
    pipeline_script_name="run_segformer_b0_pipeline.py",
    train_script_name="train_segformer_b0.py",
    evaluate_script_name="evaluate_segformer_b0.py",
    visualize_script_name="visualize_segformer_b0_predictions.py",
    output_dir=SEGFORMER_B0_OUTPUT_DIR,
    checkpoint_path=SEGFORMER_B0_CHECKPOINT_PATH,
    best_checkpoint_path=SEGFORMER_B0_BEST_CHECKPOINT_PATH,
    history_path=SEGFORMER_B0_HISTORY_PATH,
    step_history_path=SEGFORMER_B0_STEP_HISTORY_PATH,
    metrics_path=SEGFORMER_B0_METRICS_PATH,
    best_metrics_path=SEGFORMER_B0_BEST_METRICS_PATH,
    training_curves_path=SEGFORMER_B0_TRAINING_CURVES_PATH,
    step_loss_curve_path=SEGFORMER_B0_STEP_LOSS_CURVE_PATH,
    predictions_grid_path=SEGFORMER_B0_PREDICTIONS_GRID_PATH,
    encoder_weights=SEGFORMER_B0_ENCODER_WEIGHTS,
    log_every_n_batches=SEGFORMER_B0_LOG_EVERY_N_BATCHES,
    num_epochs=SEGFORMER_B0_NUM_EPOCHS,
    learning_rate=SEGFORMER_B0_LEARNING_RATE,
    optimizer=SEGFORMER_B0_OPTIMIZER,
    weight_decay=SEGFORMER_B0_WEIGHT_DECAY,
    scheduler=SEGFORMER_B0_SCHEDULER,
    loss=SEGFORMER_B0_LOSS,
)


DEEPLABV3PLUS_RESNET50_SPEC = ModelSpec(
    name="deeplabv3plus_resnet50",
    display_name="DeepLabV3+ ResNet50",
    pipeline_script_name="run_deeplabv3plus_resnet50_pipeline.py",
    train_script_name="train_deeplabv3plus_resnet50.py",
    evaluate_script_name="evaluate_deeplabv3plus_resnet50.py",
    visualize_script_name="visualize_deeplabv3plus_resnet50_predictions.py",
    output_dir=DEEPLABV3PLUS_RESNET50_OUTPUT_DIR,
    checkpoint_path=DEEPLABV3PLUS_RESNET50_CHECKPOINT_PATH,
    best_checkpoint_path=DEEPLABV3PLUS_RESNET50_BEST_CHECKPOINT_PATH,
    history_path=DEEPLABV3PLUS_RESNET50_HISTORY_PATH,
    step_history_path=DEEPLABV3PLUS_RESNET50_STEP_HISTORY_PATH,
    metrics_path=DEEPLABV3PLUS_RESNET50_METRICS_PATH,
    best_metrics_path=DEEPLABV3PLUS_RESNET50_BEST_METRICS_PATH,
    training_curves_path=DEEPLABV3PLUS_RESNET50_TRAINING_CURVES_PATH,
    step_loss_curve_path=DEEPLABV3PLUS_RESNET50_STEP_LOSS_CURVE_PATH,
    predictions_grid_path=DEEPLABV3PLUS_RESNET50_PREDICTIONS_GRID_PATH,
    encoder_weights=DEEPLABV3PLUS_RESNET50_ENCODER_WEIGHTS,
    log_every_n_batches=DEEPLABV3PLUS_RESNET50_LOG_EVERY_N_BATCHES,
    num_epochs=DEEPLABV3PLUS_RESNET50_NUM_EPOCHS,
    learning_rate=DEEPLABV3PLUS_RESNET50_LEARNING_RATE,
    optimizer=DEEPLABV3PLUS_RESNET50_OPTIMIZER,
    weight_decay=DEEPLABV3PLUS_RESNET50_WEIGHT_DECAY,
    scheduler=DEEPLABV3PLUS_RESNET50_SCHEDULER,
    loss=DEEPLABV3PLUS_RESNET50_LOSS,
)


MODEL_SPECS_BY_NAME = {
    UNET_RESNET34_SPEC.name: UNET_RESNET34_SPEC,
    DEEPLABV3PLUS_RESNET50_SPEC.name: DEEPLABV3PLUS_RESNET50_SPEC,
    SEGFORMER_B0_SPEC.name: SEGFORMER_B0_SPEC,
}


MODEL_SPECS_BY_ALIAS = {
    "unet": UNET_RESNET34_SPEC,
    "deeplab": DEEPLABV3PLUS_RESNET50_SPEC,
    "segformer": SEGFORMER_B0_SPEC,
}
