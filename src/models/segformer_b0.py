import segmentation_models_pytorch as smp


def create_segformer_b0(num_classes, encoder_weights=None):
    return smp.Segformer(
        encoder_name="mit_b0",
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
    )
