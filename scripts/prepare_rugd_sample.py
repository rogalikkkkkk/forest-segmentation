from pathlib import Path
import shutil

import numpy as np
from PIL import Image


RAW_SAMPLE_DIR = Path("data/raw/RUGD/sample/RUGD_sample-data")

RAW_IMAGES_DIR = RAW_SAMPLE_DIR / "images"
RAW_ANNOTATIONS_DIR = RAW_SAMPLE_DIR / "annotations"
COLORMAP_PATH = RAW_SAMPLE_DIR / "RUGD_annotation-colormap.txt"

PROCESSED_SAMPLE_DIR = Path("data/processed/RUGD/sample")
PROCESSED_IMAGES_DIR = PROCESSED_SAMPLE_DIR / "images"
PROCESSED_MASKS_COLOR_DIR = PROCESSED_SAMPLE_DIR / "masks_color"
PROCESSED_MASKS_ID_DIR = PROCESSED_SAMPLE_DIR / "masks_id"

IGNORE_INDEX = 255


def read_rugd_colormap(path: Path):
    """
    Читает RUGD_annotation-colormap.txt.

    Ожидаемый формат строки:
    class_id class_name R G B

    Например:
    1 dirt 108 64 20
    """
    color_to_class_id = {}
    class_id_to_name = {}

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            class_name = " ".join(parts[1:-3])
            rgb = tuple(map(int, parts[-3:]))

            color_to_class_id[rgb] = class_id
            class_id_to_name[class_id] = class_name

    return color_to_class_id, class_id_to_name


def convert_color_mask_to_id_mask(color_mask: np.ndarray, color_to_class_id: dict):
    """
    Преобразует RGB-маску в маску с ID классов.

    Вход:
        color_mask: H x W x 3

    Выход:
        id_mask: H x W, где значения пикселей — номера классов.
    """
    height, width, _ = color_mask.shape

    id_mask = np.full(
        shape=(height, width),
        fill_value=IGNORE_INDEX,
        dtype=np.uint8,
    )

    for color, class_id in color_to_class_id.items():
        color_array = np.array(color, dtype=np.uint8)
        pixels = np.all(color_mask == color_array, axis=-1)
        id_mask[pixels] = class_id

    return id_mask


def main():
    PROCESSED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_MASKS_COLOR_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_MASKS_ID_DIR.mkdir(parents=True, exist_ok=True)

    color_to_class_id, class_id_to_name = read_rugd_colormap(COLORMAP_PATH)

    print("Загруженные классы:")
    for class_id, class_name in sorted(class_id_to_name.items()):
        print(f"{class_id:02d}: {class_name}")
    print()

    image_paths = sorted(RAW_IMAGES_DIR.glob("*.png"))

    if not image_paths:
        raise RuntimeError(f"Изображения не найдены: {RAW_IMAGES_DIR}")

    print(f"Найдено изображений: {len(image_paths)}")
    print("Начинаю подготовку sample...")
    print()

    unknown_masks_count = 0

    for image_path in image_paths:
        mask_color_path = RAW_ANNOTATIONS_DIR / image_path.name

        if not mask_color_path.exists():
            raise FileNotFoundError(f"Не найдена маска для {image_path.name}")

        # 1. Копируем RGB-изображение
        processed_image_path = PROCESSED_IMAGES_DIR / image_path.name
        shutil.copy2(image_path, processed_image_path)

        # 2. Копируем цветную маску для визуальной проверки
        processed_color_mask_path = PROCESSED_MASKS_COLOR_DIR / image_path.name
        shutil.copy2(mask_color_path, processed_color_mask_path)

        # 3. Создаём маску с ID классов
        color_mask = np.array(Image.open(mask_color_path).convert("RGB"))
        id_mask = convert_color_mask_to_id_mask(color_mask, color_to_class_id)

        unique_values = np.unique(id_mask)

        if IGNORE_INDEX in unique_values:
            unknown_masks_count += 1
            print(f"Внимание: неизвестные цвета в маске {image_path.name}")

        processed_id_mask_path = PROCESSED_MASKS_ID_DIR / image_path.name
        Image.fromarray(id_mask).save(processed_id_mask_path)

    print()
    print("Подготовка завершена.")
    print(f"RGB-изображения:     {PROCESSED_IMAGES_DIR}")
    print(f"Цветные маски:       {PROCESSED_MASKS_COLOR_DIR}")
    print(f"Маски с ID классов:  {PROCESSED_MASKS_ID_DIR}")
    print(f"Масок с неизвестными цветами: {unknown_masks_count}")


if __name__ == "__main__":
    main()