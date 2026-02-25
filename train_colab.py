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
# ║  CELL 5: Validasi Model                                      ║
# ╚══════════════════════════════════════════════════════════════╝

CELL_5 = """
# --- Jalankan di Colab Cell 5 ---
from ultralytics import YOLO

# Load model terbaik dari training
model = YOLO("runs/detect/smartlab-ktm/weights/best.pt")

# Validasi
metrics = model.val()

print(f"\\nmAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")

# Print per-class results
for i, name in enumerate(model.names.values()):
    print(f"  {name}: AP50={metrics.box.ap50[i]:.3f}")
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
        ("CELL 5", "Validasi Model", CELL_5),
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
