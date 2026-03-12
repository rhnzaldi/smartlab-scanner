"""
Smart-Lab SV IPB — Text & QR Extractor
PaddleOCR untuk teks NIM/Nama, pyzbar + cv2.QRCodeDetector untuk QR Code.

Changelog v2:
  - QR: tambah upscale 2x/3x, tambah cv2.QRCodeDetector fallback, tambah CLAHE strategy
  - OCR: tambah confidence filter (score > 0.55), tambah use_angle_cls=True
  - OCR: multi-attempt (preprocessed → grayscale → color)
"""

import logging
import os
from typing import Optional

# Skip slow PaddleOCR connectivity check
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────
# Lazy-loaded singletons
# ────────────────────────────────────────────────────────
_paddle_ocr_instance = None
_paddle_ocr_init_failed = False
_pyzbar_decode = None
_pyzbar_available = None


def _get_paddle_ocr():
    """Lazy-load PaddleOCR. Dipanggil sekali, di-cache selamanya."""
    global _paddle_ocr_instance, _paddle_ocr_init_failed
    if _paddle_ocr_init_failed:
        return None
    if _paddle_ocr_instance is None:
        try:
            logger.info("⏳ Loading PaddleOCR (first time, may take a moment)...")
            from paddleocr import PaddleOCR
            _paddle_ocr_instance = PaddleOCR(
                lang="en",
                use_angle_cls=True,  # Handle teks yang sedikit miring
            )
            logger.info("✅ PaddleOCR loaded (use_angle_cls=True).")
        except Exception as e:
            logger.error(f"❌ PaddleOCR init failed: {e}")
            _paddle_ocr_init_failed = True
            return None
    return _paddle_ocr_instance


def _get_pyzbar_decode():
    """Lazy-load pyzbar. Import sekali, cache selamanya."""
    global _pyzbar_decode, _pyzbar_available
    if _pyzbar_available is False:
        return None
    if _pyzbar_decode is None:
        try:
            from pyzbar.pyzbar import decode as pyzbar_decode
            _pyzbar_decode = pyzbar_decode
            _pyzbar_available = True
        except ImportError:
            logger.error("pyzbar not installed. Run: pip install pyzbar")
            _pyzbar_available = False
            return None
    return _pyzbar_decode


# ────────────────────────────────────────────────────────
# QR Code Extraction
# ────────────────────────────────────────────────────────

def _try_cv2_qr_decode(img: np.ndarray) -> Optional[str]:
    """
    Fallback QR decode menggunakan cv2.QRCodeDetector (built-in OpenCV, no extra dep).
    Bekerja pada gambar grayscale atau color BGR.
    """
    try:
        detector = cv2.QRCodeDetector()
        data, _, _ = detector.detectAndDecode(img)
        if data:
            return data.strip()
    except Exception:
        pass
    return None


def extract_qr(img: np.ndarray) -> Optional[str]:
    """
    Decode QR code dari image crop.

    Strategi (dicoba berurutan, berhenti saat salah satu berhasil):
      1. Grayscale asli (sudah diupscale dari pipeline)
      2. OTSU binary
      3. OTSU inverted
      4. Upscale 2× + OTSU binary
      5. Upscale 3× + OTSU binary
      6. CLAHE + OTSU (untuk QR dengan kontras rendah)
      7. cv2.QRCodeDetector fallback (grayscale & binary)

    Input: bisa grayscale atau BGR — fungsi ini normalize sendiri.
    """
    pyzbar_decode = _get_pyzbar_decode()

    # Normalize ke grayscale dulu
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Hitung threshold OTSU sekali pakai
    _, binary     = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_inv    = cv2.bitwise_not(binary)

    # Upscale variants
    h, w = gray.shape
    gray_2x  = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    gray_3x  = cv2.resize(gray, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    _, bin_2x = cv2.threshold(gray_2x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bin_3x = cv2.threshold(gray_3x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # CLAHE variant (normalisasi kontras lokal)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray_clahe = clahe.apply(gray)
    _, bin_clahe = cv2.threshold(gray_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    strategies = [
        ("gray",          gray),
        ("binary",        binary),
        ("binary_inv",    binary_inv),
        ("2x_binary",     bin_2x),
        ("3x_binary",     bin_3x),
        ("clahe_binary",  bin_clahe),
    ]

    # ── pyzbar strategies ──
    if pyzbar_decode is not None:
        try:
            from pyzbar.pyzbar import ZBarSymbol
            for name, processed in strategies:
                results = pyzbar_decode(processed, symbols=[ZBarSymbol.QRCODE])
                if results:
                    data = results[0].data.decode("utf-8", errors="ignore").strip()
                    if data:
                        logger.info(f"QR decoded via pyzbar [{name}]: '{data}'")
                        return data
        except Exception as e:
            logger.debug(f"pyzbar error: {e}")

    # ── cv2.QRCodeDetector fallback ──
    for name, processed in [("cv2_gray", gray), ("cv2_binary", binary), ("cv2_2x", gray_2x)]:
        data = _try_cv2_qr_decode(processed)
        if data:
            logger.info(f"QR decoded via cv2 [{name}]: '{data}'")
            return data

    logger.warning("⚠️ QR code could not be decoded with any strategy.")
    return None


# ────────────────────────────────────────────────────────
# OCR Text Extraction
# ────────────────────────────────────────────────────────

_OCR_CONFIDENCE_THRESHOLD = 0.55  # Abaikan teks dengan confidence < ini


def _run_ocr_on(img: np.ndarray) -> Optional[str]:
    """
    Jalankan PaddleOCR pada satu image (grayscale atau BGR).
    Filter baris teks dengan confidence < _OCR_CONFIDENCE_THRESHOLD.
    Kembalikan teks gabungan atau None jika kosong.
    """
    ocr = _get_paddle_ocr()
    if ocr is None:
        return None

    h, w = img.shape[:2]
    if h < 16 or w < 32:
        return None

    try:
        results = ocr.predict(img)
        if not results:
            return None

        texts = []
        for result_item in results:
            rec_texts  = result_item.get("rec_texts", [])
            rec_scores = result_item.get("rec_scores", [])
            for text, score in zip(rec_texts, rec_scores):
                if score >= _OCR_CONFIDENCE_THRESHOLD:
                    texts.append(text)
                    logger.debug(f"  OCR line accepted: '{text}' (conf: {score:.2f})")
                else:
                    logger.debug(f"  OCR line rejected (low conf {score:.2f}): '{text}'")

        combined = " ".join(texts).strip()
        return combined if combined else None

    except Exception as e:
        logger.debug(f"  OCR error: {e}")
        return None


def extract_text_ocr(img: np.ndarray) -> Optional[str]:
    """
    Ekstrak teks dari image crop menggunakan PaddleOCR.

    Multi-attempt (fallback jika attempt sebelumnya gagal):
      1. Input yang diterima (biasanya sudah preprocessed: CLAHE + sharpen)
      2. Grayscale mentah (tanpa preprocessing tambahan)
      3. Image color asli BGR (PaddleOCR kadang lebih baik dengan color)

    Returns: teks gabungan semua baris, atau None.
    """
    if img is None or img.size == 0:
        return None

    h, w = img.shape[:2]
    if h < 16 or w < 32:
        logger.debug(f"  Image too small for OCR: {w}x{h}")
        return None

    # Attempt 1: gunakan img yang sudah masuk (sudah preprocessed dari pipeline)
    result = _run_ocr_on(img)
    if result:
        logger.info(f"OCR result (attempt 1): '{result}'")
        return result

    # Attempt 2: grayscale saja (tanpa CLAHE/sharpen)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Upscale jika perlu untuk grayscale fallback
    gh, gw = gray.shape
    if gh < 64:
        scale = 64 / gh
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    result = _run_ocr_on(gray)
    if result:
        logger.info(f"OCR result (attempt 2 gray): '{result}'")
        return result

    logger.warning("⚠️ OCR returned no confident text after all attempts.")
    return None
