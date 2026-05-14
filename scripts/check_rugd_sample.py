from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


RUGD_SAMPLE_DIR = Path("data/raw/RUGD/sample/RUGD_sample-data")

IMAGES_DIR = RUGD_SAMPLE_DIR / "images"
ANNOTATIONS_DIR = RUGD_SAMPLE_DIR / "annotations"
COLORMAP_PATH = RUGD_SAMPLE_DIR / "RUGD_annotation-colormap.txt"

OUTPUT_DIR = Path("outputs/dataset_check")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_colormap(path: Path):
    """
    Пытается прочитать файл colormap.
    Ожидаемый смысл строк: название_класса R G B
    """
    classes = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) < 4:
                continue

            try:
                r, g, b = map(int, parts[-3:])
                class_name = " ".join(parts[:-3])
                classes.append((class_name, (r, g, b)))
            except ValueError:
                continue

    return classes


def get_unique_colors(mask: np.ndarray):
    """
    Возвращает уникальные RGB-цвета из маски.
    """
    if mask.ndim != 3 or mask.shape[2] != 3:
        raise ValueError("Ожидалась цветная RGB-маска размера H x W x 3")

    colors = np.unique(mask.reshape(-1, 3), axis=0)
    return [tuple(color.tolist()) for color in colors]


def main():
    print("Проверка RUGD sample")
    print("=" * 60)

    if not RUGD_SAMPLE_DIR.exists():
        raise FileNotFoundError(f"Папка не найдена: {RUGD_SAMPLE_DIR}")

    image_paths = sorted(IMAGES_DIR.glob("*.png"))
    annotation_paths = sorted(ANNOTATIONS_DIR.glob("*.png"))

    print(f"Папка с изображениями: {IMAGES_DIR}")
    print(f"Папка с аннотациями:   {ANNOTATIONS_DIR}")
    print(f"Файл colormap:         {COLORMAP_PATH}")
    print()

    print(f"Количество изображений: {len(image_paths)}")
    print(f"Количество масок:       {len(annotation_paths)}")
    print()

    image_names = {path.name for path in image_paths}
    annotation_names = {path.name for path in annotation_paths}

    missing_annotations = image_names - annotation_names
    missing_images = annotation_names - image_names

    print(f"Изображений без маски: {len(missing_annotations)}")
    print(f"Масок без изображения: {len(missing_images)}")
    print()

    if missing_annotations:
        print("Примеры изображений без маски:")
        print(list(sorted(missing_annotations))[:5])

    if missing_images:
        print("Примеры масок без изображения:")
        print(list(sorted(missing_images))[:5])

    classes = read_colormap(COLORMAP_PATH)

    print("Классы из RUGD_annotation-colormap.txt:")
    for idx, (class_name, color) in enumerate(classes):
        print(f"{idx:02d}: {class_name:20s} {color}")
    print()

    if not image_paths:
        raise RuntimeError("Изображения не найдены")

    first_image_path = image_paths[0]
    first_mask_path = ANNOTATIONS_DIR / first_image_path.name

    image = np.array(Image.open(first_image_path).convert("RGB"))
    mask = np.array(Image.open(first_mask_path).convert("RGB"))

    print("Проверка первого примера:")
    print(f"Изображение: {first_image_path.name}")
    print(f"Размер изображения: {image.shape}")
    print(f"Размер маски:       {mask.shape}")
    print()

    unique_colors = get_unique_colors(mask)

    print(f"Количество уникальных цветов в первой маске: {len(unique_colors)}")
    print("Уникальные цвета в первой маске:")
    for color in unique_colors:
        print(color)

    # Создаём изображение для быстрой визуальной проверки
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(image)
    axes[0].set_title("RGB image")
    axes[0].axis("off")

    axes[1].imshow(mask)
    axes[1].set_title("Segmentation mask")
    axes[1].axis("off")

    plt.tight_layout()

    output_path = OUTPUT_DIR / "rugd_sample_check.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print()
    print(f"Картинка для проверки сохранена: {output_path}")


if __name__ == "__main__":
    main()