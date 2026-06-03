from config import RUGD_NUM_CLASSES
from model_specs import ModelSpec
from models.deeplabv3plus_resnet50 import create_deeplabv3plus_resnet50
from models.segformer_b0 import create_segformer_b0
from models.unet_resnet34 import UNetResNet34


def create_model(model_spec: ModelSpec):
    if model_spec.name == "unet_resnet34":
        return UNetResNet34(
            num_classes=RUGD_NUM_CLASSES,
            encoder_weights=model_spec.encoder_weights,
        )

    if model_spec.name == "segformer_b0":
        return create_segformer_b0(
            num_classes=RUGD_NUM_CLASSES,
            encoder_weights=model_spec.encoder_weights,
        )

    if model_spec.name == "deeplabv3plus_resnet50":
        return create_deeplabv3plus_resnet50(
            num_classes=RUGD_NUM_CLASSES,
            encoder_weights=model_spec.encoder_weights,
        )

    raise ValueError(f"Unsupported model: {model_spec.name}")
