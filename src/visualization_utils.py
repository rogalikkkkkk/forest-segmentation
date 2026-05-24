from pathlib import Path
import csv
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def read_id_to_color(path):
    id_to_color = {}

    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()
            class_id = int(parts[0])
            rgb = tuple(map(int, parts[-3:]))
            id_to_color[class_id] = rgb

    return id_to_color


def denormalize_image(image_tensor, image_mean, image_std):
    image = image_tensor.detach().cpu().numpy()

    mean = np.array(image_mean, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(image_std, dtype=np.float32).reshape(3, 1, 1)

    image = image * std + mean
    image = np.clip(image, 0.0, 1.0)
    image = np.transpose(image, (1, 2, 0))

    return image


def colorize_mask(mask, id_to_color):
    height, width = mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for class_id, color in id_to_color.items():
        color_mask[mask == class_id] = color

    return color_mask


def select_fixed_random_indices(dataset_size, sample_count, seed):
    rng = random.Random(seed)
    sample_count = min(sample_count, dataset_size)
    return sorted(rng.sample(range(dataset_size), sample_count))


def save_selected_samples(dataset, indices, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["index", "filename"])

        for index in indices:
            writer.writerow([index, dataset[index]["filename"]])


def save_epoch_prediction_grid(
    model,
    dataset,
    indices,
    device,
    output_path,
    id_to_color,
    image_mean,
    image_std,
    title,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()

    fig, axes = plt.subplots(
        len(indices),
        3,
        figsize=(12, 4 * len(indices)),
    )

    if len(indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    with torch.inference_mode():
        for row, sample_index in enumerate(indices):
            sample = dataset[sample_index]

            image = sample["image"].unsqueeze(0).to(device)
            mask = sample["mask"].numpy()
            filename = sample["filename"]

            logits = model(image)
            prediction = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

            image_for_plot = denormalize_image(sample["image"], image_mean, image_std)
            mask_color = colorize_mask(mask, id_to_color)
            prediction_color = colorize_mask(prediction, id_to_color)

            axes[row, 0].imshow(image_for_plot)
            axes[row, 0].set_title(f"Image\n{filename}")
            axes[row, 0].axis("off")

            axes[row, 1].imshow(mask_color)
            axes[row, 1].set_title("Ground truth")
            axes[row, 1].axis("off")

            axes[row, 2].imshow(prediction_color)
            axes[row, 2].set_title("Prediction")
            axes[row, 2].axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    if was_training:
        model.train()
