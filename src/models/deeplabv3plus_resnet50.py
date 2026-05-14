import segmentation_models_pytorch as smp


def create_deeplabv3plus_resnet50(
    num_classes,
    encoder_weights=None,
):
    return smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
    )
