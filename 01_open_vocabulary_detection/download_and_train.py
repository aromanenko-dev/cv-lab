"""
Download a tiger-labeled dataset from Roboflow and fine-tune YOLO26 on it.

This is a small, fast demonstration fine-tune (few epochs, nano model) meant to prove
the concept -- not a production-grade training run. Solution 3 in the tiger vocabulary
problem notebook.

Setup:
    pip install roboflow ultralytics

You'll need a free Roboflow API key (https://roboflow.com > Settings > API Keys).

Usage:
    export ROBOFLOW_API_KEY="your_key_here"
    python download_and_train.py
"""

import os
import random
import shutil
from pathlib import Path

from roboflow import Roboflow
from ultralytics import YOLO

API_KEY = os.environ.get("ROBOFLOW_API_KEY")

# --- Step 1: download the dataset ---
rf = Roboflow(api_key=API_KEY)
project = rf.workspace("trackabox-4ejy9").project("tiger-ahgqa")
version = project.version(1)  # verify this is the current version on the Universe page
dataset = version.download("yolov8")

print(f"Dataset downloaded to: {dataset.location}")

# --- Step 1b: split train into train/val (Roboflow export omits valid/) ---
root = Path(dataset.location)
train_images_dir = root / "train" / "images"
train_labels_dir = root / "train" / "labels"
val_images_dir = root / "valid" / "images"
val_labels_dir = root / "valid" / "labels"
val_images_dir.mkdir(parents=True, exist_ok=True)
val_labels_dir.mkdir(parents=True, exist_ok=True)

image_files = sorted(train_images_dir.glob("*"))
random.seed(42)
random.shuffle(image_files)

n_val = max(1, int(len(image_files) * 0.15))
val_files = image_files[:n_val]

print(f"Total images: {len(image_files)}")
print(f"Moving {n_val} images (15%) to a new validation split...")

for img_path in val_files:
    label_path = train_labels_dir / (img_path.stem + ".txt")
    shutil.move(str(img_path), str(val_images_dir / img_path.name))
    if label_path.exists():
        shutil.move(str(label_path), str(val_labels_dir / label_path.name))

print(f"Train: {len(image_files) - n_val} images | Val: {n_val} images")

# --- Step 2: fine-tune YOLO26 nano on it ---
model = YOLO("yolo26n.pt")  # start from COCO-pretrained weights, not from scratch

results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=10,          # small demo run; a real run would use 100+
    imgsz=640,
    device="mps",        # switch to "cpu" or "cuda" as needed
    project="runs_tiger",
    name="finetune_demo",
)

print("\nTraining complete.")
print(f"Best weights saved to: {results.save_dir}/weights/best.pt")
