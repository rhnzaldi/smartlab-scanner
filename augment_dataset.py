#!/usr/bin/env python3
"""
Smart-Lab SV IPB — Offline Dataset Augmentor
==============================================
Augmentasi SEMUA split (train, valid, test) termasuk transformasi
bounding box YOLO. Jalankan di Google Colab SETELAH download dataset Roboflow.

Menggunakan OpenCV murni (tanpa library augmentasi tambahan).

Usage (di Colab):
    !python augment_dataset.py --dataset "/content/dataset-name-1" --multiply-train 4 --multiply-val 6 --multiply-test 6
"""

import argparse
import glob
import math
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


# ────────────────────────────────────────────────────────
# YOLO BBox Helpers
# ────────────────────────────────────────────────────────
def parse_yolo_labels(label_path: str) -> List[Tuple[int, float, float, float, float]]:
    """Parse YOLO format label file → list of (class_id, cx, cy, w, h)."""
    labels = []
    if not os.path.exists(label_path):
        return labels
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                labels.append((cls_id, cx, cy, w, h))
    return labels


def save_yolo_labels(label_path: str, labels: List[Tuple[int, float, float, float, float]]):
    """Save labels ke file YOLO format."""
    with open(label_path, "w") as f:
        for cls_id, cx, cy, w, h in labels:
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def clamp_bbox(cx, cy, w, h) -> Tuple[float, float, float, float]:
    """Clamp bounding box agar tetap di dalam [0, 1]."""
    x1 = max(0.0, cx - w / 2)
    y1 = max(0.0, cy - h / 2)
    x2 = min(1.0, cx + w / 2)
    y2 = min(1.0, cy + h / 2)

    new_w = x2 - x1
    new_h = y2 - y1
    new_cx = x1 + new_w / 2
    new_cy = y1 + new_h / 2

    return new_cx, new_cy, new_w, new_h


# ────────────────────────────────────────────────────────
# Augmentation Functions (image + bbox aware)
# ────────────────────────────────────────────────────────

def aug_brightness(img: np.ndarray, labels, *, delta_range=(-40, 40)):
    """Ubah brightness secara random."""
    delta = random.randint(*delta_range)
    img = np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)
    return img, labels  # bbox tidak berubah


def aug_contrast(img: np.ndarray, labels, *, alpha_range=(0.6, 1.4)):
    """Ubah contrast secara random."""
    alpha = random.uniform(*alpha_range)
    img = np.clip((img.astype(np.float32) * alpha), 0, 255).astype(np.uint8)
    return img, labels


def aug_saturation(img: np.ndarray, labels, *, factor_range=(0.5, 1.5)):
    """Ubah saturation secara random."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    factor = random.uniform(*factor_range)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return img, labels


def aug_hue(img: np.ndarray, labels, *, delta_range=(-15, 15)):
    """Shift hue secara random."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    delta = random.randint(*delta_range)
    hsv[:, :, 0] = (hsv[:, :, 0] + delta) % 180
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return img, labels


def aug_gaussian_noise(img: np.ndarray, labels, *, sigma_range=(5, 25)):
    """Tambahkan Gaussian noise."""
    sigma = random.randint(*sigma_range)
    noise = np.random.normal(0, sigma, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img, labels


def aug_blur(img: np.ndarray, labels, *, ksize_choices=(3, 5)):
    """Gaussian blur ringan."""
    k = random.choice(ksize_choices)
    img = cv2.GaussianBlur(img, (k, k), 0)
    return img, labels


def aug_horizontal_flip(img: np.ndarray, labels):
    """Flip horizontal + transform bbox."""
    img = cv2.flip(img, 1)
    new_labels = []
    for cls_id, cx, cy, w, h in labels:
        new_cx = 1.0 - cx
        new_labels.append((cls_id, new_cx, cy, w, h))
    return img, new_labels


def aug_rotation(img: np.ndarray, labels, *, angle_range=(-15, 15)):
    """
    Rotasi kecil + transform bbox.
    Menggunakan affine transformation untuk bbox.
    """
    h, w = img.shape[:2]
    angle = random.uniform(*angle_range)

    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img_rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    new_labels = []
    rad = math.radians(-angle)  # OpenCV uses counter-clockwise
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    for cls_id, cx, cy, bw, bh in labels:
        # Convert normalized to pixel
        px = cx * w
        py = cy * h

        # Rotate around center
        dx = px - w / 2
        dy = py - h / 2
        new_px = cos_a * dx - sin_a * dy + w / 2
        new_py = sin_a * dx + cos_a * dy + h / 2

        # Convert back to normalized
        new_cx = new_px / w
        new_cy = new_py / h

        # Slightly expand bbox to account for rotation
        expand = 1.0 + abs(math.sin(2 * rad)) * 0.15
        new_bw = min(bw * expand, 1.0)
        new_bh = min(bh * expand, 1.0)

        new_cx, new_cy, new_bw, new_bh = clamp_bbox(new_cx, new_cy, new_bw, new_bh)

        # Skip if bbox became too small
        if new_bw > 0.01 and new_bh > 0.01:
            new_labels.append((cls_id, new_cx, new_cy, new_bw, new_bh))

    return img_rotated, new_labels


def aug_scale_crop(img: np.ndarray, labels, *, scale_range=(0.8, 1.0)):
    """
    Random crop & scale — simulasi zoom in/out.
    Menghapus bbox yang keluar dari crop area.
    """
    h, w = img.shape[:2]
    scale = random.uniform(*scale_range)

    # Crop area
    crop_w = int(w * scale)
    crop_h = int(h * scale)
    x_off = random.randint(0, w - crop_w)
    y_off = random.randint(0, h - crop_h)

    img_cropped = img[y_off:y_off + crop_h, x_off:x_off + crop_w]
    img_resized = cv2.resize(img_cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    # Transform labels
    new_labels = []
    for cls_id, cx, cy, bw, bh in labels:
        # Convert to pixel coordinates in original
        px = cx * w
        py = cy * h
        pw = bw * w
        ph = bh * h

        # Shift by crop offset
        new_px = px - x_off
        new_py = py - y_off

        # Convert to new normalized coords
        new_cx = new_px / crop_w
        new_cy = new_py / crop_h
        new_bw = pw / crop_w
        new_bh = ph / crop_h

        new_cx, new_cy, new_bw, new_bh = clamp_bbox(new_cx, new_cy, new_bw, new_bh)

        # Skip if center is outside or bbox too small
        if 0.0 < new_cx < 1.0 and 0.0 < new_cy < 1.0 and new_bw > 0.02 and new_bh > 0.02:
            new_labels.append((cls_id, new_cx, new_cy, new_bw, new_bh))

    return img_resized, new_labels


def aug_perspective(img: np.ndarray, labels, *, strength=0.03):
    """
    Random perspective warp ringan — simulasi sudut kamera berbeda.
    Untuk dataset kecil, ini SANGAT penting karena webcam posisinya bisa bervariasi.
    """
    h, w = img.shape[:2]

    # Random perspective points
    d = strength
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [random.uniform(0, w * d), random.uniform(0, h * d)],
        [w - random.uniform(0, w * d), random.uniform(0, h * d)],
        [w - random.uniform(0, w * d), h - random.uniform(0, h * d)],
        [random.uniform(0, w * d), h - random.uniform(0, h * d)],
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    img_warped = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # Transform bbox centers through perspective matrix
    new_labels = []
    for cls_id, cx, cy, bw, bh in labels:
        px = cx * w
        py = cy * h

        # Transform center point
        pt = np.float32([[[px, py]]])
        new_pt = cv2.perspectiveTransform(pt, M)[0][0]

        new_cx = new_pt[0] / w
        new_cy = new_pt[1] / h

        # Bbox size roughly stays the same for small warps
        new_cx, new_cy, bw, bh = clamp_bbox(new_cx, new_cy, bw, bh)

        if bw > 0.01 and bh > 0.01:
            new_labels.append((cls_id, new_cx, new_cy, bw, bh))

    return img_warped, new_labels


# ────────────────────────────────────────────────────────
# Augmentation Pipeline
# ────────────────────────────────────────────────────────

# Augmentasi yang TIDAK mengubah bbox (aman digabung banyak)
COLOR_AUGS = [
    aug_brightness,
    aug_contrast,
    aug_saturation,
    aug_hue,
    aug_gaussian_noise,
    aug_blur,
]

# Augmentasi yang MENGUBAH bbox (max 1-2 per gambar)
SPATIAL_AUGS = [
    aug_horizontal_flip,
    aug_rotation,
    aug_scale_crop,
    aug_perspective,
]


def apply_random_augmentations(
    img: np.ndarray,
    labels: List[Tuple[int, float, float, float, float]],
    num_color: int = 3,
    num_spatial: int = 1,
) -> Tuple[np.ndarray, List]:
    """
    Apply random combination of augmentations.
    - num_color: jumlah color augmentation yang digabung
    - num_spatial: jumlah spatial augmentation (max 1-2, karena bbox bisa drift)
    """
    # Apply color augmentations
    chosen_color = random.sample(COLOR_AUGS, min(num_color, len(COLOR_AUGS)))
    for aug_fn in chosen_color:
        img, labels = aug_fn(img, labels)

    # Apply spatial augmentations
    if num_spatial > 0:
        chosen_spatial = random.sample(SPATIAL_AUGS, min(num_spatial, len(SPATIAL_AUGS)))
        for aug_fn in chosen_spatial:
            img, labels = aug_fn(img, labels)

    return img, labels


# ────────────────────────────────────────────────────────
# Main Augmentor
# ────────────────────────────────────────────────────────
def augment_split(
    images_dir: str,
    labels_dir: str,
    multiplier: int,
    split_name: str,
):
    """
    Augment satu split (train/valid/test).
    Untuk setiap gambar asli, generate `multiplier` variasi baru.
    """
    image_files = sorted(
        glob.glob(os.path.join(images_dir, "*.jpg"))
        + glob.glob(os.path.join(images_dir, "*.jpeg"))
        + glob.glob(os.path.join(images_dir, "*.png"))
    )

    if not image_files:
        print(f"  ⚠️ No images found in {images_dir}")
        return

    original_count = len(image_files)
    generated = 0

    print(f"\n⏳ [{split_name}] Augmenting {original_count} images × {multiplier}x ...", end="", flush=True)

    for img_path in image_files:
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"  ⚠️ Cannot read: {img_path}")
            continue

        # Load labels
        basename = Path(img_path).stem
        label_path = os.path.join(labels_dir, f"{basename}.txt")
        labels = parse_yolo_labels(label_path)

        # Generate augmented versions
        for i in range(multiplier):
            # Vary augmentation intensity
            num_color = random.randint(2, 4)
            num_spatial = random.randint(0, 2)

            aug_img, aug_labels = apply_random_augmentations(
                img.copy(),
                labels.copy(),
                num_color=num_color,
                num_spatial=num_spatial,
            )

            # Skip if all labels were lost during spatial augmentation
            if labels and not aug_labels:
                # Retry with color-only
                aug_img, aug_labels = apply_random_augmentations(
                    img.copy(), labels.copy(),
                    num_color=4, num_spatial=0,
                )

            # Save augmented image
            aug_name = f"{basename}_aug{i:03d}"
            aug_img_path = os.path.join(images_dir, f"{aug_name}.jpg")
            aug_label_path = os.path.join(labels_dir, f"{aug_name}.txt")

            cv2.imwrite(aug_img_path, aug_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            save_yolo_labels(aug_label_path, aug_labels)
            generated += 1

        # Tampilkan titik setiap 10 gambar selesai diproses
        idx = image_files.index(img_path) + 1
        if idx % 10 == 0 or idx == original_count:
            print(f" {idx}/{original_count}", end="", flush=True)

    print(f"\n✅ [{split_name}] {original_count} originals + {generated} augmented = {original_count + generated} total")


def find_dataset_structure(dataset_path: str) -> dict:
    """
    Auto-detect struktur folder dataset YOLO.
    Support beberapa format umum dari Roboflow:
    - dataset/train/images, dataset/train/labels
    - dataset/images/train, dataset/labels/train
    """
    splits = {}

    for split_name in ["train", "valid", "test"]:
        # Format 1: dataset/split/images & dataset/split/labels
        img_dir = os.path.join(dataset_path, split_name, "images")
        lbl_dir = os.path.join(dataset_path, split_name, "labels")
        if os.path.isdir(img_dir) and os.path.isdir(lbl_dir):
            splits[split_name] = {"images": img_dir, "labels": lbl_dir}
            continue

        # Format 2: dataset/images/split & dataset/labels/split
        img_dir = os.path.join(dataset_path, "images", split_name)
        lbl_dir = os.path.join(dataset_path, "labels", split_name)
        if os.path.isdir(img_dir) and os.path.isdir(lbl_dir):
            splits[split_name] = {"images": img_dir, "labels": lbl_dir}
            continue

    return splits


def main():
    parser = argparse.ArgumentParser(
        description="Augment ALL splits of a YOLO dataset (train, valid, test)"
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path ke root folder dataset YOLO (dari Roboflow download)"
    )
    parser.add_argument(
        "--multiply-train", type=int, default=4,
        help="Jumlah variasi per gambar di TRAIN set (default: 4)"
    )
    parser.add_argument(
        "--multiply-val", type=int, default=6,
        help="Jumlah variasi per gambar di VALID set (default: 6)"
    )
    parser.add_argument(
        "--multiply-test", type=int, default=6,
        help="Jumlah variasi per gambar di TEST set (default: 6)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed untuk reproducibility"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("🔧 Smart-Lab Dataset Augmentor")
    print(f"   Dataset: {args.dataset}")
    print(f"   Multipliers: train={args.multiply_train}x, val={args.multiply_val}x, test={args.multiply_test}x")

    # Detect structure
    splits = find_dataset_structure(args.dataset)

    if not splits:
        print(f"\n❌ Cannot detect dataset structure in: {args.dataset}")
        print("   Expected: train/images, train/labels or images/train, labels/train")
        sys.exit(1)

    print(f"\n   Detected splits: {list(splits.keys())}")

    # Count before
    print("\n📊 BEFORE augmentation:")
    for name, dirs in splits.items():
        count = len(glob.glob(os.path.join(dirs["images"], "*.*")))
        print(f"   {name}: {count} images")

    # Augment each split
    multipliers = {
        "train": args.multiply_train,
        "valid": args.multiply_val,
        "test": args.multiply_test,
    }

    for split_name, dirs in splits.items():
        mult = multipliers.get(split_name, 4)
        if mult > 0:
            augment_split(
                images_dir=dirs["images"],
                labels_dir=dirs["labels"],
                multiplier=mult,
                split_name=split_name,
            )

    # Count after
    print("\n" + "=" * 50)
    print("📊 AFTER augmentation:")
    total = 0
    for name, dirs in splits.items():
        count = len(glob.glob(os.path.join(dirs["images"], "*.*")))
        total += count
        print(f"   {name}: {count} images")
    print(f"   TOTAL: {total} images")
    print("=" * 50)
    print("\n✅ Augmentation complete! Dataset siap untuk training.")


if __name__ == "__main__":
    main()
