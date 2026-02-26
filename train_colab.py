#!/usr/bin/env python3
"""
Smart-Lab SV IPB — YOLOv8 Training Script (untuk Google Colab)

CARA PAKAI:
===========
1. Buka Google Colab (https://colab.research.google.com)
2. Buat notebook baru
3. Pastikan runtime GPU aktif: Runtime → Change runtime type → GPU (T4)
4. Copy-paste cell-cell di bawah ini ke notebook Colab

CATATAN:
========
- Script ini BUKAN untuk dijalankan secara langsung di laptop
- Script ini adalah PANDUAN cell-by-cell untuk Google Colab
- Setelah training selesai, download file best.pt dan copy ke models/
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 1: Install Dependencies                                ║
# ╚══════════════════════════════════════════════════════════════╝

CELL_1 = """
# --- Jalankan di Colab Cell 1 ---
!pip install ultralytics roboflow
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 2: Check GPU                                           ║
# ╚══════════════════════════════════════════════════════════════╝

CELL_2 = """
# --- Jalankan di Colab Cell 2 ---
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only!'}")

# PENTING: Jika output 'CPU only!', ganti runtime ke GPU dulu!
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 3: Download Dataset dari Roboflow                      ║
# ╚══════════════════════════════════════════════════════════════╝

CELL_3 = """
# --- Jalankan di Colab Cell 3 ---
# GANTI dengan API key dan project Anda dari Roboflow!

from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")  # ← GANTI INI
project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT")  # ← GANTI INI
version = project.version(1)  # ← GANTI nomor version jika perlu
dataset = version.download("yolov8")

print(f"Dataset downloaded to: {dataset.location}")
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 3B: AUGMENTASI DATASET (PENTING!)                      ║
# ║  Memperbanyak valid & test set yang terlalu sedikit           ║
# ╚══════════════════════════════════════════════════════════════╝

CELL_3B = """
# --- Jalankan di Colab Cell 3B ---
# Upload augment_dataset.py dari ScanKtm/ ke Colab:
from google.colab import files
uploaded = files.upload()  # ← pilih file augment_dataset.py

# Jalankan augmentasi:
# - Train: x4 (50 → 250)
# - Valid: x6 (2 → 14) 
# - Test:  x6 (2 → 14)
# Total: ~278 images (dari 54 asli)
!python augment_dataset.py --dataset "{dataset.location}" --multiply-train 4 --multiply-val 6 --multiply-test 6

# TIPS: Jika di Roboflow Anda sudah ubah split ke 70/15/15,
# maka valid & test sudah lebih banyak, dan hasil akhirnya:
# Train: ~26 × 5 = ~130, Valid: ~8 × 7 = ~56, Test: ~8 × 7 = ~56
# Total: ~242 images yang jauh lebih sehat!
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 4: Training YOLOv8                                     ║
# ╚══════════════════════════════════════════════════════════════╝

CELL_4 = """
# --- Jalankan di Colab Cell 4 ---
from ultralytics import YOLO

# Load base model (yolov8n = nano, paling ringan dan cepat)
model = YOLO("yolov8n.pt")

# Training
# TIPS untuk dataset kecil (85 gambar):
# - epochs tinggi (150-300) karena data sedikit
# - patience tinggi (50) agar tidak early-stop terlalu cepat
# - imgsz 640 standar untuk YOLOv8
# - batch 16 atau 8 tergantung GPU memory

results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=200,           # Banyak epoch karena data sedikit
    imgsz=640,
    batch=16,
    patience=50,          # Toleransi early stopping
    device=0,             # GPU
    workers=2,
    name="smartlab-ktm",

    # Augmentasi AGRESIF untuk dataset kecil
    hsv_h=0.015,          # Hue shift
    hsv_s=0.7,            # Saturation
    hsv_v=0.4,            # Value/brightness
    degrees=10,           # Rotation ±10°
    translate=0.1,        # Translation ±10%
    scale=0.5,            # Scale ±50%
    shear=5,              # Shear ±5°
    flipud=0.0,           # JANGAN flip vertikal (kartu tidak boleh terbalik)
    fliplr=0.5,           # Flip horizontal OK
    mosaic=1.0,           # Mosaic augmentation
    mixup=0.1,            # Mixup augmentation
    copy_paste=0.1,       # Copy-paste augmentation
)

print("Training complete!")
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 5: Evaluasi Model (Metrics Lengkap untuk Laporan)     ║
# ╚══════════════════════════════════════════════════════════════╝

CELL_5 = """
# --- Jalankan di Colab Cell 5 ---
# Evaluasi model dengan metrik lengkap untuk laporan proyek

from ultralytics import YOLO
import json

model = YOLO("runs/detect/smartlab-ktm/weights/best.pt")

# ── Validasi pada dataset test/valid ──
metrics = model.val(split="test")  # ganti "test" → "val" jika test set kosong

# ═══════════════════════════════════════════
# METRIK UTAMA (untuk laporan)
# ═══════════════════════════════════════════
print("=" * 60)
print("     EVALUASI MODEL YOLOv8 — Smart-Lab KTM Scanner")
print("=" * 60)

# Overall metrics
print(f"\\n📊 OVERALL METRICS:")
print(f"  Precision (P):     {metrics.box.mp:.4f}  ({metrics.box.mp:.1%})")
print(f"  Recall (R):        {metrics.box.mr:.4f}  ({metrics.box.mr:.1%})")
print(f"  mAP@50:            {metrics.box.map50:.4f}  ({metrics.box.map50:.1%})")
print(f"  mAP@50-95:         {metrics.box.map:.4f}  ({metrics.box.map:.1%})")

# F1 Score (harmonic mean of P and R)
p = metrics.box.mp
r = metrics.box.mr
f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
print(f"  F1 Score:          {f1:.4f}  ({f1:.1%})")

# Per-class metrics
class_names = list(model.names.values())
print(f"\\n📋 PER-CLASS METRICS:")
print(f"  {'Class':<15} {'Precision':>10} {'Recall':>10} {'AP@50':>10} {'AP@50-95':>10}")
print(f"  {'-'*55}")

for i, name in enumerate(class_names):
    ap50 = metrics.box.ap50[i] if i < len(metrics.box.ap50) else 0
    ap = metrics.box.ap[i] if i < len(metrics.box.ap) else 0
    # Per-class P and R from metrics
    pc = metrics.box.p[i] if i < len(metrics.box.p) else 0
    rc = metrics.box.r[i] if i < len(metrics.box.r) else 0
    print(f"  {name:<15} {pc:>10.4f} {rc:>10.4f} {ap50:>10.4f} {ap:>10.4f}")

print(f"\\n  {'OVERALL':<15} {p:>10.4f} {r:>10.4f} {metrics.box.map50:>10.4f} {metrics.box.map:>10.4f}")

# Speed metrics
print(f"\\n⚡ SPEED (per image):")
print(f"  Preprocessing:     {metrics.speed['preprocess']:.1f}ms")
print(f"  Inference:         {metrics.speed['inference']:.1f}ms")
print(f"  Postprocessing:    {metrics.speed['postprocess']:.1f}ms")
total_speed = sum(metrics.speed.values())
print(f"  Total:             {total_speed:.1f}ms ({1000/total_speed:.0f} FPS)")

print(f"\\n" + "=" * 60)
print(f"  F1 Score = {f1:.4f} | mAP@50 = {metrics.box.map50:.4f}")
print("=" * 60)
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 5B: Visualisasi & Confusion Matrix                    ║
# ╚══════════════════════════════════════════════════════════════╝

CELL_5B = """
# --- Jalankan di Colab Cell 5B ---
# Generate confusion matrix dan visualisasi training curves

import matplotlib.pyplot as plt
from matplotlib.image import imread
import os
import glob

train_dir = "runs/detect/smartlab-ktm"

# ── Confusion Matrix ──
cm_path = os.path.join(train_dir, "confusion_matrix_normalized.png")
if os.path.exists(cm_path):
    plt.figure(figsize=(10, 8))
    plt.imshow(imread(cm_path))
    plt.axis("off")
    plt.title("Confusion Matrix (Normalized)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    print("✅ Confusion Matrix — simpan untuk laporan!")
else:
    print("⚠️ Confusion matrix tidak ditemukan di:", cm_path)

# ── Training Curves ──
curves = [
    ("results.png", "Training Results (Loss, Precision, Recall, mAP)"),
    ("F1_curve.png", "F1-Confidence Curve"),
    ("PR_curve.png", "Precision-Recall Curve"),
    ("P_curve.png", "Precision-Confidence Curve"),
    ("R_curve.png", "Recall-Confidence Curve"),
]

for fname, title in curves:
    fpath = os.path.join(train_dir, fname)
    if os.path.exists(fpath):
        plt.figure(figsize=(12, 6))
        plt.imshow(imread(fpath))
        plt.axis("off")
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    else:
        print(f"⚠️ {fname} tidak ditemukan")

# ── Download semua plot sebagai ZIP ──
import shutil
plots_dir = "/content/evaluation_plots"
os.makedirs(plots_dir, exist_ok=True)

for fname, _ in curves:
    src = os.path.join(train_dir, fname)
    if os.path.exists(src):
        shutil.copy(src, plots_dir)

# Copy confusion matrices
for cm in ["confusion_matrix.png", "confusion_matrix_normalized.png"]:
    src = os.path.join(train_dir, cm)
    if os.path.exists(src):
        shutil.copy(src, plots_dir)

# Zip and download
shutil.make_archive("/content/evaluation_plots", "zip", plots_dir)
from google.colab import files
files.download("/content/evaluation_plots.zip")
print("\\n✅ Semua plot evaluasi di-download sebagai evaluation_plots.zip")
print("   Gunakan untuk laporan proyek!")
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 5C: Tabel Ringkasan (Copy-Paste ke Laporan)          ║
# ╚══════════════════════════════════════════════════════════════╝

CELL_5C = """
# --- Jalankan di Colab Cell 5C ---
# Generate tabel ringkasan yang siap copy-paste ke laporan

from ultralytics import YOLO

model = YOLO("runs/detect/smartlab-ktm/weights/best.pt")
metrics = model.val(split="test")

class_names = list(model.names.values())
p = metrics.box.mp
r = metrics.box.mr
f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0

print("\\n" + "=" * 70)
print("TABEL UNTUK LAPORAN (format Markdown — bisa paste ke Word/Docs)")
print("=" * 70)

# Tabel 1: Overall
print("\\n### Tabel 1: Hasil Evaluasi Model YOLOv8 (Overall)")
print("| Metrik | Nilai |")
print("|--------|-------|")
print(f"| Precision | {p:.4f} ({p:.1%}) |")
print(f"| Recall | {r:.4f} ({r:.1%}) |")
print(f"| F1 Score | {f1:.4f} ({f1:.1%}) |")
print(f"| mAP@50 | {metrics.box.map50:.4f} ({metrics.box.map50:.1%}) |")
print(f"| mAP@50-95 | {metrics.box.map:.4f} ({metrics.box.map:.1%}) |")
total_speed = sum(metrics.speed.values())
print(f"| Inference Speed | {metrics.speed['inference']:.1f}ms/image |")

# Tabel 2: Per-Class
print("\\n### Tabel 2: Evaluasi Per-Kelas Objek")
print("| Kelas | Precision | Recall | F1 | AP@50 |")
print("|-------|-----------|--------|----|-------|")
for i, name in enumerate(class_names):
    pc = metrics.box.p[i] if i < len(metrics.box.p) else 0
    rc = metrics.box.r[i] if i < len(metrics.box.r) else 0
    f1c = 2 * (pc * rc) / (pc + rc) if (pc + rc) > 0 else 0
    ap50 = metrics.box.ap50[i] if i < len(metrics.box.ap50) else 0
    print(f"| {name} | {pc:.4f} | {rc:.4f} | {f1c:.4f} | {ap50:.4f} |")

# Tabel 3: Konfigurasi Training
print("\\n### Tabel 3: Konfigurasi Training")
print("| Parameter | Nilai |")
print("|-----------|-------|")
print("| Model | YOLOv8n (Nano) |")
print("| Pre-trained | COCO dataset |")
print("| Epochs | 200 |")
print("| Image Size | 640×640 |")
print("| Batch Size | 16 |")
print("| Optimizer | AdamW |")
print("| Learning Rate | 0.01 (auto) |")
print("| Augmentation | Mosaic, Mixup, HSV, Flip, Rotation |")

print("\\n" + "=" * 70)
print("✅ Copy tabel di atas ke laporan proyek Anda!")
print("=" * 70)
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 6: Test Prediksi                                       ║
# ╚══════════════════════════════════════════════════════════════╝

CELL_6 = """
# --- Jalankan di Colab Cell 6 ---
from ultralytics import YOLO
from google.colab import files
import cv2
from matplotlib import pyplot as plt

model = YOLO("runs/detect/smartlab-ktm/weights/best.pt")

# Upload gambar test
uploaded = files.upload()
filename = list(uploaded.keys())[0]

# Prediksi
results = model.predict(filename, conf=0.35)

# Tampilkan
result_img = results[0].plot()
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("KTM Detection Result")
plt.show()

# Print detections
for box in results[0].boxes:
    cls_name = model.names[int(box.cls[0])]
    conf = float(box.conf[0])
    print(f"  Detected: {cls_name} ({conf:.1%})")
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 7: Download Model                                      ║
# ╚══════════════════════════════════════════════════════════════╝

CELL_7 = """
# --- Jalankan di Colab Cell 7 ---
from google.colab import files

# Download best.pt ke laptop Anda
files.download("runs/detect/smartlab-ktm/weights/best.pt")

print("\\n✅ SELESAI! Copy file best.pt yang ter-download ke folder:")
print("   ScanKtm/models/best.pt")
print("\\nLalu jalankan: python test_webcam.py")
"""


# ────────────────────────────────────────────────────────
# Print guide jika dijalankan langsung
# ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Smart-Lab SV IPB — YOLOv8 Training Guide")
    print("=" * 60)
    print()
    print("Script ini BUKAN untuk dijalankan di laptop!")
    print("Buka Google Colab dan copy-paste cell-cell berikut:")
    print()

    cells = [
        ("CELL 1", "Install Dependencies", CELL_1),
        ("CELL 2", "Check GPU", CELL_2),
        ("CELL 3", "Download Dataset (Roboflow)", CELL_3),
        ("CELL 3B", "⭐ AUGMENTASI DATASET (valid & test)", CELL_3B),
        ("CELL 4", "Training YOLOv8", CELL_4),
        ("CELL 5", "📊 Evaluasi Model (Metrik Lengkap)", CELL_5),
        ("CELL 5B", "📈 Confusion Matrix & Training Curves", CELL_5B),
        ("CELL 5C", "📋 Tabel Ringkasan (Copy ke Laporan)", CELL_5C),
        ("CELL 6", "Test Prediksi", CELL_6),
        ("CELL 7", "Download Model", CELL_7),
    ]

    for name, desc, code in cells:
        print(f"── {name}: {desc} ──")
        print(code.strip())
        print()

    print("=" * 60)
    print("Setelah training, copy best.pt ke: models/best.pt")
    print("Lalu jalankan: python test_webcam.py")
    print("=" * 60)
