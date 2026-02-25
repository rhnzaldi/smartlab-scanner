"""
Smart-Lab SV IPB — Image Preprocessor
OpenCV preprocessing utilities untuk meningkatkan akurasi OCR pada crop KTM.
"""

import cv2
import numpy as np


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert BGR image ke grayscale. Jika sudah grayscale, return as-is."""
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_blur(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Gaussian Blur untuk mengurangi noise.
    ksize harus ganjil (3, 5, 7).
    """
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def apply_sharpen(img: np.ndarray) -> np.ndarray:
    """
    Sharpen image menggunakan unsharp mask kernel.
    Meningkatkan ketajaman teks agar OCR lebih akurat.
    """
    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel)


def apply_threshold(img: np.ndarray) -> np.ndarray:
    """
    Adaptive threshold — menghasilkan biner hitam-putih.
    Sangat efektif untuk teks di KTM dengan background berwarna.
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


def resize_if_small(img: np.ndarray, min_height: int = 50) -> np.ndarray:
    """
    Upscale crop yang terlalu kecil. OCR kesulitan membaca teks < 50px.
    """
    h, w = img.shape[:2]
    if h < min_height:
        scale = min_height / h
        new_w = int(w * scale)
        img = cv2.resize(img, (new_w, min_height), interpolation=cv2.INTER_CUBIC)
    return img


def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    """
    Full preprocessing chain untuk OCR:
    1. Resize jika terlalu kecil
    2. Grayscale
    3. Gaussian Blur (noise reduction)
    4. Sharpen (ketajaman teks)

    Tidak menggunakan threshold di chain utama karena
    PaddleOCR biasanya lebih baik dengan grayscale daripada biner.
    """
    img = resize_if_small(img)
    img = to_grayscale(img)
    img = apply_blur(img, ksize=3)
    img = apply_sharpen(img)
    return img


def preprocess_for_qr(img: np.ndarray) -> np.ndarray:
    """
    Preprocessing khusus untuk QR code decoding.
    QR decoder butuh kontras tinggi.
    """
    img = resize_if_small(img, min_height=100)
    gray = to_grayscale(img)
    # OTSU threshold memberikan biner sharp untuk QR
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary
