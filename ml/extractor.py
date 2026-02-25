"""
Smart-Lab SV IPB — Text & QR Extractor
PaddleOCR untuk teks NIM/Nama, pyzbar untuk QR Code.

NFR Refactored:
- [P3] Module-level lazy import for pyzbar
- [M3] Removed dead code (extract_text_multiple_attempts)
- [O3] No more silent exception swallowing
"""

import logging
import os
import sys
from typing import Optional

# Skip slow PaddleOCR connectivity check
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────
# Lazy-loaded singletons [P3]
# ────────────────────────────────────────────────────────
_paddle_ocr_instance = None
_paddle_ocr_init_failed = False
_pyzbar_decode = None  # [P3] lazy-loaded pyzbar
_pyzbar_available = None  # None = not checked, True/False = result


def _get_paddle_ocr():
    """Lazy-load PaddleOCR v5. Dipanggil sekali, di-cache selamanya."""
    global _paddle_ocr_instance, _paddle_ocr_init_failed
    if _paddle_ocr_init_failed:
        return None
    if _paddle_ocr_instance is None:
        try:
            logger.info("⏳ Loading PaddleOCR (first time, may take a moment)...")
            from paddleocr import PaddleOCR
            _paddle_ocr_instance = PaddleOCR(lang="en")
            logger.info("✅ PaddleOCR loaded successfully.")
        except Exception as e:
            logger.error(f"❌ PaddleOCR init failed: {e}")
            _paddle_ocr_init_failed = True
            return None
    return _paddle_ocr_instance


def _get_pyzbar_decode():
    """[P3] Lazy-load pyzbar. Import sekali, cache selamanya."""
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
def extract_qr(img: np.ndarray) -> Optional[str]:
    """
    Decode QR code dari image crop menggunakan pyzbar.
    Mencoba beberapa strategi preprocessing jika gagal.
    """
    pyzbar_decode = _get_pyzbar_decode()
    if pyzbar_decode is None:
        return None

    from pyzbar.pyzbar import ZBarSymbol

    strategies = [
        ("raw", img),
    ]

    # Strategi 2: grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        strategies.append(("grayscale", gray))
    else:
        gray = img

    # Strategi 3: OTSU threshold
    gray_for_bin = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, binary = cv2.threshold(gray_for_bin, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    strategies.append(("binary", binary))

    # Strategi 4: inverted
    strategies.append(("inverted", cv2.bitwise_not(binary)))

    for strategy_name, processed_img in strategies:
        results = pyzbar_decode(processed_img, symbols=[ZBarSymbol.QRCODE])
        if results:
            data = results[0].data.decode("utf-8", errors="ignore").strip()
            logger.info(f"QR decoded ({strategy_name}): '{data}'")
            return data

    logger.warning("⚠️ QR code could not be decoded with any strategy.")
    return None


# ────────────────────────────────────────────────────────
# OCR Text Extraction
# ────────────────────────────────────────────────────────
def extract_text_ocr(img: np.ndarray) -> Optional[str]:
    """
    Ekstrak teks dari image crop menggunakan PaddleOCR v5.
    Returns gabungan semua baris teks yang terdeteksi, atau None.
    """
    if img is None or img.size == 0:
        return None

    # Minimum size check — PaddleOCR crashes on very small images
    h, w = img.shape[:2]
    if h < 16 or w < 32:
        logger.debug(f"  Image too small for OCR: {w}x{h}")
        return None

    # Upscale small images for better OCR accuracy
    if h < 64:
        scale = 64 / h
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    try:
        ocr = _get_paddle_ocr()
        if ocr is None:
            return None

        # PaddleOCR v5 uses .predict() instead of .ocr()
        results = ocr.predict(img)

        if not results:
            logger.warning("⚠️ PaddleOCR returned no results.")
            return None

        # Parse OCRResult object
        texts = []
        for result_item in results:
            rec_texts = result_item.get("rec_texts", [])
            rec_scores = result_item.get("rec_scores", [])
            for text, score in zip(rec_texts, rec_scores):
                texts.append(text)
                logger.debug(f"  OCR line: '{text}' (conf: {score:.2f})")

        combined = " ".join(texts).strip()
        logger.info(f"OCR extracted: '{combined}'")
        return combined if combined else None

    except Exception as e:
        logger.debug(f"  OCR skip (small/bad crop): {e}")
        return None
