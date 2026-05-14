from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


PROCESSED_SAMPLE_DIR = Path("data/processed/RUGD/sample")

IMAGES_DIR = PROCESSED_SAMPLE_DIR / "images"
MASKS_COLOR_DIR = PROCESSED_SAMPLE_DIR / "masks_color"
MASKS_ID_DIR = PROCESSED_SAMPLE_DIR / "masks_id"

OUTPUT_DIR = Path("outputs/dataset_check")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    image_paths = sorted(IMAGES_DIR.glob("*.png"))
    mask_id_paths = sorted(MASKS_ID_DIR.glob("*.png"))

    print(f"Количество изображений: {len(image_paths)}")
    print(f"Количество ID-масок:    {len(mask_id_paths)}")
    print()

    if len(image_paths) != len(mask_id_paths):
        raise ValueError("Количество изображений и ID-масок не совпадает")

    all_values = set()

    for mask_path in mask_id_paths:
        mask = np.array(Image.open(mask_path))
        unique_values = np.unique(mask)

        print(f"{mask_path.name}: {unique_values.tolist()}")

        for value in unique_values:
            all_values.add(int(value))

    print()
    print(f"Все найденные значения классов: {sorted(all_values)}")

    first_image_path = image_paths[0]
    first_mask_color_path = MASKS_COLOR_DIR / first_image_path.name
    first_mask_id_path = MASKS_ID_DIR / first_image_path.name

    image = np.array(Image.open(first_image_path).convert("RGB"))
    mask_color = np.array(Image.open(first_mask_color_path).convert("RGB"))
    mask_id = np.array(Image.open(first_mask_id_path))

    print()
    print("Первый пример:")
    print(f"Изображение: {first_image_path.name}")
    print(f"Размер image:      {image.shape}")
    print(f"Размер mask_color: {mask_color.shape}")
    print(f"Размер mask_id:    {mask_id.shape}")
    print(f"Значения mask_id:  {np.unique(mask_id).tolist()}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("RGB image")
    axes[0].axis("off")

    axes[1].imshow(mask_color)
    axes[1].set_title("Color mask")
    axes[1].axis("off")

    axes[2].imshow(mask_id)
    axes[2].set_title("ID mask")
    axes[2].axis("off")

    plt.tight_layout()

    output_path = OUTPUT_DIR / "rugd_processed_check.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print()
    print(f"Проверочное изображение сохранено: {output_path}")


if __name__ == "__main__":
    main()