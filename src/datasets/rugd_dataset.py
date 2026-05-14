from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class RUGDDataset(Dataset):
    def __init__(
        self,
        images_dir,
        masks_dir,
        image_height=256,
        image_width=320,
        image_mean=None,
        image_std=None,
        split_file=None,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)

        self.image_height = image_height
        self.image_width = image_width

        self.image_mean = np.array(
            image_mean if image_mean is not None else [0.485, 0.456, 0.406],
            dtype=np.float32,
        )

        self.image_std = np.array(
            image_std if image_std is not None else [0.229, 0.224, 0.225],
            dtype=np.float32,
        )

        self.image_paths = self._collect_image_paths(split_file)

        if not self.image_paths:
            raise RuntimeError(f"No images found: {self.images_dir}")

        self.mask_paths = []

        for image_path in self.image_paths:
            mask_path = self.masks_dir / image_path.name

            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Mask not found for image {image_path.name}: {mask_path}"
                )

            self.mask_paths.append(mask_path)

    def _collect_image_paths(self, split_file):
        if split_file is None:
            return sorted(self.images_dir.glob("*.png"))

        split_path = Path(split_file)

        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")

        filenames = [
            line.strip()
            for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        return [self.images_dir / filename for filename in filenames]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        image = image.resize(
            (self.image_width, self.image_height),
            resample=Image.BILINEAR,
        )

        # Segmentation masks must use nearest neighbor resize to preserve class IDs.
        mask = mask.resize(
            (self.image_width, self.image_height),
            resample=Image.NEAREST,
        )

        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.int64)

        image = (image - self.image_mean) / self.image_std

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()

        return {
            "image": image,
            "mask": mask,
            "filename": image_path.name,
        }
