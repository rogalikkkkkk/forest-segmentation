from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from config import (
    RUGD_SAMPLE_IMAGES_DIR,
    RUGD_SAMPLE_MASKS_ID_DIR,
    RUGD_NUM_CLASSES,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    BATCH_SIZE,
    IMAGE_MEAN,
    IMAGE_STD,
    DATASET_CHECK_OUTPUT_DIR,
)
from datasets.rugd_dataset import RUGDDataset


def denormalize_image(image_tensor):
    """
    Преобразует нормализованный tensor C x H x W обратно в обычное изображение H x W x C.
    """
    image = image_tensor.numpy()

    mean = np.array(IMAGE_MEAN).reshape(3, 1, 1)
    std = np.array(IMAGE_STD).reshape(3, 1, 1)

    image = image * std + mean
    image = np.clip(image, 0, 1)

    image = np.transpose(image, (1, 2, 0))

    return image


def main():
    dataset = RUGDDataset(
        images_dir=RUGD_SAMPLE_IMAGES_DIR,
        masks_dir=RUGD_SAMPLE_MASKS_ID_DIR,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        image_mean=IMAGE_MEAN,
        image_std=IMAGE_STD,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    print("Проверка RUGD Dataset и DataLoader")
    print("=" * 60)
    print(f"Количество примеров в dataset: {len(dataset)}")
    print(f"Размер изображения: {IMAGE_HEIGHT} x {IMAGE_WIDTH}")
    print(f"Количество классов RUGD: {RUGD_NUM_CLASSES}")
    print()

    batch = next(iter(dataloader))

    images = batch["image"]
    masks = batch["mask"]
    filenames = batch["filename"]

    print(f"Batch image shape: {images.shape}")
    print(f"Batch mask shape:  {masks.shape}")
    print(f"Filenames: {filenames}")
    print()

    print(f"Тип image tensor: {images.dtype}")
    print(f"Тип mask tensor:  {masks.dtype}")
    print()

    print(f"Минимальное значение в image: {images.min().item():.4f}")
    print(f"Максимальное значение в image: {images.max().item():.4f}")
    print()

    first_mask = masks[0].numpy()
    unique_values = np.unique(first_mask)

    print(f"Уникальные классы в первой маске batch:")
    print(unique_values.tolist())
    print()

    invalid_values = [
        int(value)
        for value in unique_values
        if value < 0 or value >= RUGD_NUM_CLASSES
    ]

    if invalid_values:
        print("ОШИБКА: найдены некорректные значения классов:")
        print(invalid_values)
    else:
        print("Все значения классов корректны.")

    image_for_plot = denormalize_image(images[0])
    mask_for_plot = masks[0].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(image_for_plot)
    axes[0].set_title(f"Image: {filenames[0]}")
    axes[0].axis("off")

    axes[1].imshow(mask_for_plot, vmin=0, vmax=RUGD_NUM_CLASSES - 1)
    axes[1].set_title("ID mask after resize")
    axes[1].axis("off")

    plt.tight_layout()

    output_path = DATASET_CHECK_OUTPUT_DIR / "rugd_dataloader_check.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print()
    print(f"Проверочное изображение сохранено: {output_path}")


if __name__ == "__main__":
    main()
