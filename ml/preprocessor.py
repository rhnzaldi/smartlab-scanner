"""
Smart-Lab SV IPB — Image Preprocessor
OpenCV preprocessing utilities untuk meningkatkan akurasi OCR dan QR decode pada crop KTM.

Changelog:
  v2 — Tambah CLAHE, deskew, naikkan min_height untuk OCR
"""

import cv2
import numpy as np


# ────────────────────────────────────────────────────────
# Primitives
# ────────────────────────────────────────────────────────

def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert BGR image ke grayscale. Jika sudah grayscale, return as-is."""
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_blur(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Gaussian Blur untuk mengurangi noise kamera. ksize harus ganjil (3, 5, 7)."""
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def apply_sharpen(img: np.ndarray) -> np.ndarray:
    """
    Sharpen image menggunakan unsharp mask kernel.
    Meningkatkan ketajaman tepi teks agar OCR lebih akurat.
    """
    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0],
    ], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel)


def apply_threshold(img: np.ndarray) -> np.ndarray:
    """
    Adaptive threshold — menghasilkan biner hitam-putih.
    Efektif untuk teks di KTM dengan background berwarna/tidak rata.
    """
    if len(img.shape) == 3:
        img = to_grayscale(img)
    return cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Normalisasi kontras lokal — sangat membantu untuk:
      - KTM dengan pencahayaan tidak merata (shadow/highlight)
      - Foto KTM yang under/over-exposed
      - Teks pada area background gelap atau terang

    Input harus grayscale.
    """
    if len(img.shape) == 3:
        img = to_grayscale(img)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(img)


def deskew(img: np.ndarray, max_angle: float = 10.0) -> np.ndarray:
    """
    Koreksi kemiringan gambar (deskewing) hingga ±max_angle derajat.
    Memanfaatkan distribusi piksel horizontal untuk mendeteksi sudut kemiringan.
    Hanya aktif jika angle terdeteksi signifikan (>0.5°) agar tidak distorsi gambar lurus.

    Input: grayscale image.
    """
    if len(img.shape) == 3:
        img = to_grayscale(img)

    h, w = img.shape

    # Binerisasi dulu untuk mendeteksi sudut dari tepi teks
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Cari koordinat non-zero pixels
    coords = np.column_stack(np.where(binary > 0))

    if len(coords) < 50:
        # Terlalu sedikit piksel untuk deskew yang akurat
        return img

    # minAreaRect menggunakan konveks hull → kemiringan dominan
    angle = cv2.minAreaRect(coords)[-1]

    # Standardize angle
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    # Hanya rotasi jika kemiringan signifikan
    if abs(angle) < 0.5 or abs(angle) > max_angle:
        return img

    # Rotasi di sekitar pusat gambar
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def resize_for_ocr(img: np.ndarray, min_height: int = 100) -> np.ndarray:
    """
    Upscale crop yang terlalu kecil untuk OCR.
    PaddleOCR membutuhkan minimal ~64px,  min 100px untuk akurasi optimal.
    Aspect ratio dijaga.
    """
    h, w = img.shape[:2]
    if h < min_height:
        scale = min_height / h
        new_w = max(1, int(w * scale))
        img = cv2.resize(img, (new_w, min_height), interpolation=cv2.INTER_CUBIC)
    return img


def resize_for_qr(img: np.ndarray, min_size: int = 200) -> np.ndarray:
    """
    Upscale crop QR code jika terlalu kecil.
    pyzbar / cv2.QRCodeDetector butuh minimal ~100px, ideal 200px+ per sisi.
    """
    h, w = img.shape[:2]
    min_dim = min(h, w)
    if min_dim < min_size:
        scale = min_size / min_dim
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return img


# ────────────────────────────────────────────────────────
# High-Level Chains
# ────────────────────────────────────────────────────────

def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    """
    Full preprocessing chain untuk OCR teks (nama/NIM):
      1. Upscale ke min 100px tinggi
      2. Grayscale
      3. CLAHE  — normalisasi kontras adaptif
      4. Blur   — noise reduction
      5. Sharpen — ketajaman tepi teks

    PaddleOCR dijalankan dengan input grayscale (lebih konsisten dari color).
    Adaptive threshold TIDAK digunakan karena PaddleOCR bekerja lebih baik tanpa biner.
    """
    img = resize_for_ocr(img, min_height=100)
    img = to_grayscale(img)
    img = apply_clahe(img)
    img = apply_blur(img, ksize=3)
    img = apply_sharpen(img)
    return img


def preprocess_for_qr(img: np.ndarray) -> np.ndarray:
    """
    Preprocessing minimal untuk QR code — hanya upscale dan grayscale.
    Strategi binerisasi ditangani oleh extract_qr() sendiri agar tidak konflik.
    """
    img = resize_for_qr(img, min_size=200)
    img = to_grayscale(img)
    return img
