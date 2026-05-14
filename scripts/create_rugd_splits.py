from pathlib import Path
import random
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from config import (
    RUGD_IMAGES_DIR,
    RUGD_MASKS_ID_DIR,
    RUGD_TEST_SPLIT_PATH,
    RUGD_TRAIN_SPLIT_PATH,
    RUGD_VAL_SPLIT_PATH,
)


TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
SEED = 42


def main():
    image_paths = sorted(RUGD_IMAGES_DIR.glob("*.png"))

    if not image_paths:
        raise RuntimeError(f"No images found: {RUGD_IMAGES_DIR}")

    filenames = []
    missing_masks = []

    for image_path in image_paths:
        mask_path = RUGD_MASKS_ID_DIR / image_path.name

        if not mask_path.exists():
            missing_masks.append(image_path.name)
            continue

        filenames.append(image_path.name)

    if missing_masks:
        raise RuntimeError(
            f"Found {len(missing_masks)} images without masks. "
            f"First missing mask: {missing_masks[0]}"
        )

    random_generator = random.Random(SEED)
    random_generator.shuffle(filenames)

    total_count = len(filenames)
    train_count = int(total_count * TRAIN_RATIO)
    val_count = int(total_count * VAL_RATIO)

    train_filenames = sorted(filenames[:train_count])
    val_filenames = sorted(filenames[train_count : train_count + val_count])
    test_filenames = sorted(filenames[train_count + val_count :])

    RUGD_TRAIN_SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)

    RUGD_TRAIN_SPLIT_PATH.write_text(
        "\n".join(train_filenames) + "\n",
        encoding="utf-8",
    )
    RUGD_VAL_SPLIT_PATH.write_text(
        "\n".join(val_filenames) + "\n",
        encoding="utf-8",
    )
    RUGD_TEST_SPLIT_PATH.write_text(
        "\n".join(test_filenames) + "\n",
        encoding="utf-8",
    )

    print("RUGD train/val/test split created")
    print("=" * 60)
    print(f"Images dir: {RUGD_IMAGES_DIR}")
    print(f"Masks dir: {RUGD_MASKS_ID_DIR}")
    print(f"Total pairs: {total_count}")
    print(f"Train ratio: {TRAIN_RATIO}")
    print(f"Val ratio: {VAL_RATIO}")
    print(f"Test ratio: {TEST_RATIO}")
    print(f"Train size: {len(train_filenames)}")
    print(f"Val size: {len(val_filenames)}")
    print(f"Test size: {len(test_filenames)}")
    print(f"Seed: {SEED}")
    print(f"Train split: {RUGD_TRAIN_SPLIT_PATH}")
    print(f"Val split: {RUGD_VAL_SPLIT_PATH}")
    print(f"Test split: {RUGD_TEST_SPLIT_PATH}")


if __name__ == "__main__":
    main()
