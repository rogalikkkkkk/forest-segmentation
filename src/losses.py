from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


SUPPORTED_LOSSES = ("ce", "weighted_ce")


def load_class_weights(class_weights_path, device):
    path = Path(class_weights_path)

    if not path.exists():
        raise FileNotFoundError(
            "Class weights file not found. "
            f"Run scripts/compute_rugd_class_weights.py first: {path}"
        )

    weights = np.load(path)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def create_loss(loss_name, class_weights_path=None, device=None):
    if loss_name == "ce":
        return nn.CrossEntropyLoss()

    if loss_name == "weighted_ce":
        if class_weights_path is None:
            raise ValueError("class_weights_path is required for weighted_ce")

        weights = load_class_weights(class_weights_path, device=device)
        return nn.CrossEntropyLoss(weight=weights)

    raise ValueError(
        f"Unsupported loss: {loss_name}. Supported values: {SUPPORTED_LOSSES}"
    )
