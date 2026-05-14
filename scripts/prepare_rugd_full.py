from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_RUGD_DIR = PROJECT_ROOT / "data" / "raw" / "RUGD"
RAW_IMAGES_DIR = RAW_RUGD_DIR / "frames"
RAW_ANNOTATIONS_DIR = RAW_RUGD_DIR / "annotations"
COLORMAP_PATH = (
    RAW_RUGD_DIR / "sample" / "RUGD_sample-data" / "RUGD_annotation-colormap.txt"
)

PROCESSED_RUGD_DIR = PROJECT_ROOT / "data" / "processed" / "RUGD"
PROCESSED_IMAGES_DIR = PROCESSED_RUGD_DIR / "images"
PROCESSED_MASKS_COLOR_DIR = PROCESSED_RUGD_DIR / "masks_color"
PROCESSED_MASKS_ID_DIR = PROCESSED_RUGD_DIR / "masks_id"

IGNORE_INDEX = 255
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def read_rugd_colormap(path):
    color_to_class_id = {}
    class_id_to_name = {}

    with path.open("r", encoding="utf-8") as file:
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


def build_color_lookup(color_to_class_id):
    lookup = np.full(256 * 256 * 256, IGNORE_INDEX, dtype=np.uint8)

    for (red, green, blue), class_id in color_to_class_id.items():
        color_key = red * 256 * 256 + green * 256 + blue
        lookup[color_key] = class_id

    return lookup


def convert_color_mask_to_id_mask(color_mask, color_lookup):
    color_mask = color_mask.astype(np.int32)
    color_keys = (
        color_mask[:, :, 0] * 256 * 256
        + color_mask[:, :, 1] * 256
        + color_mask[:, :, 2]
    )

    return color_lookup[color_keys]


def collect_files(root_dir, extensions):
    return sorted(
        path
        for path in root_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def build_annotation_index(annotation_paths):
    by_name = {}
    by_stem = {}

    for annotation_path in annotation_paths:
        by_name[annotation_path.name] = annotation_path
        by_stem[annotation_path.stem] = annotation_path

    return by_name, by_stem


def find_annotation_for_image(image_path, annotations_by_name, annotations_by_stem):
    if image_path.name in annotations_by_name:
        return annotations_by_name[image_path.name]

    return annotations_by_stem.get(image_path.stem)


def main():
    if not RAW_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Raw RUGD images directory not found: {RAW_IMAGES_DIR}")

    if not RAW_ANNOTATIONS_DIR.exists():
        raise FileNotFoundError(
            f"Raw RUGD annotations directory not found: {RAW_ANNOTATIONS_DIR}"
        )

    if not COLORMAP_PATH.exists():
        raise FileNotFoundError(f"RUGD colormap not found: {COLORMAP_PATH}")

    color_to_class_id, class_id_to_name = read_rugd_colormap(COLORMAP_PATH)
    color_lookup = build_color_lookup(color_to_class_id)

    image_paths = collect_files(RAW_IMAGES_DIR, IMAGE_EXTENSIONS)
    annotation_paths = collect_files(RAW_ANNOTATIONS_DIR, IMAGE_EXTENSIONS)

    print("Preparing full RUGD")
    print("=" * 60)
    print(f"Raw images dir: {RAW_IMAGES_DIR}")
    print(f"Raw annotations dir: {RAW_ANNOTATIONS_DIR}")
    print(f"Found raw images: {len(image_paths)}")
    print(f"Found raw annotations: {len(annotation_paths)}")
    print(f"Loaded classes: {len(class_id_to_name)}")
    print()

    if not image_paths:
        raise RuntimeError(
            "No raw images found. Put full RUGD frames into "
            f"{RAW_IMAGES_DIR} before running this script."
        )

    if not annotation_paths:
        raise RuntimeError(
            "No raw annotations found. Put full RUGD annotations into "
            f"{RAW_ANNOTATIONS_DIR} before running this script."
        )

    PROCESSED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_MASKS_COLOR_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_MASKS_ID_DIR.mkdir(parents=True, exist_ok=True)

    annotations_by_name, annotations_by_stem = build_annotation_index(annotation_paths)

    prepared_count = 0
    skipped_count = 0
    missing_annotations = []
    unknown_color_masks = []

    for image_path in image_paths:
        annotation_path = find_annotation_for_image(
            image_path=image_path,
            annotations_by_name=annotations_by_name,
            annotations_by_stem=annotations_by_stem,
        )

        if annotation_path is None:
            missing_annotations.append(image_path.name)
            continue

        output_name = f"{image_path.stem}.png"
        processed_image_path = PROCESSED_IMAGES_DIR / output_name
        processed_color_mask_path = PROCESSED_MASKS_COLOR_DIR / output_name
        processed_id_mask_path = PROCESSED_MASKS_ID_DIR / output_name

        if (
            processed_image_path.exists()
            and processed_color_mask_path.exists()
            and processed_id_mask_path.exists()
        ):
            skipped_count += 1
            prepared_count += 1
            continue

        if not processed_image_path.exists():
            image = Image.open(image_path).convert("RGB")
            image.save(processed_image_path)

        color_mask_image = Image.open(annotation_path).convert("RGB")
        if not processed_color_mask_path.exists():
            color_mask_image.save(processed_color_mask_path)

        color_mask = np.array(color_mask_image)
        id_mask = convert_color_mask_to_id_mask(color_mask, color_lookup)

        if IGNORE_INDEX in np.unique(id_mask):
            unknown_color_masks.append(annotation_path.name)

        Image.fromarray(id_mask).save(processed_id_mask_path)
        prepared_count += 1

    print("Preparation finished.")
    print(f"Prepared pairs: {prepared_count}")
    print(f"Skipped existing pairs: {skipped_count}")
    print(f"Missing annotations: {len(missing_annotations)}")
    print(f"Masks with unknown colors: {len(unknown_color_masks)}")
    print(f"Processed images dir: {PROCESSED_IMAGES_DIR}")
    print(f"Processed color masks dir: {PROCESSED_MASKS_COLOR_DIR}")
    print(f"Processed ID masks dir: {PROCESSED_MASKS_ID_DIR}")

    if missing_annotations:
        print()
        print("First missing annotations:")
        for filename in missing_annotations[:10]:
            print(f"  {filename}")

    if unknown_color_masks:
        print()
        print("First masks with unknown colors:")
        for filename in unknown_color_masks[:10]:
            print(f"  {filename}")


if __name__ == "__main__":
    main()
