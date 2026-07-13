"""
Roboflow's export for this dataset version only contains a `train` split, even though
`data.yaml` references `valid` and `test` folders that don't exist. This script creates a
real train/val split from the existing images so validation isn't performed on training data.

Usage:
    python split_dataset.py /path/to/dataset
"""

import argparse
import random
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=str, help="Path to the dataset root (contains train/)")
    parser.add_argument("--val_fraction", type=float, default=0.15, help="Fraction of data to hold out for validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.dataset_dir)
    train_images_dir = root / "train" / "images"
    train_labels_dir = root / "train" / "labels"

    val_images_dir = root / "valid" / "images"
    val_labels_dir = root / "valid" / "labels"
    val_images_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(train_images_dir.glob("*"))
    random.seed(args.seed)
    random.shuffle(image_files)

    n_val = max(1, int(len(image_files) * args.val_fraction))
    val_files = image_files[:n_val]

    print(f"Total images: {len(image_files)}")
    print(f"Moving {n_val} images ({args.val_fraction:.0%}) to a new validation split...")

    for img_path in val_files:
        label_path = train_labels_dir / (img_path.stem + ".txt")

        shutil.move(str(img_path), str(val_images_dir / img_path.name))
        if label_path.exists():
            shutil.move(str(label_path), str(val_labels_dir / label_path.name))

    print(f"Done. Train now has {len(image_files) - n_val} images, val has {n_val}.")
    print(f"Update data.yaml's `val:` line to `../valid/images` if it isn't already.")


if __name__ == "__main__":
    main()
