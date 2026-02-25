"""
Smart-Lab SV IPB — Data Validator
Regex cleaning dan NIM double-validation.
"""

import re
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────
# NIM Cleaning
# ────────────────────────────────────────────────────────

# [M2] Separated: digit corrections vs chars to strip
_OCR_TO_DIGIT = {
    "I": "1", "l": "1", "|": "1",   # I/l/pipe → 1
    "O": "0", "o": "0",             # O/o → 0
    "S": "5", "s": "5",             # S/s → 5
    "Z": "2", "z": "2",             # Z/z → 2
    "B": "8",                        # B → 8
    "G": "6", "g": "9",             # G → 6, g → 9
    "T": "7",                        # T → 7
    "A": "4",                        # A → 4
}

_STRIP_CHARS = set(" .,-")  # chars to remove from NIM


def clean_nim(raw: Optional[str]) -> Optional[str]:
    """
    Bersihkan raw OCR output dari NIM.

    Format NIM IPB: bisa berupa:
    - J0403231061 (J + 10 digit)
    - J3B121015   (J3B + 6 digit)
    - atau variasi lain

    Strategi:
    1. Cari prefix "J" diikuti digit-digit
    2. Bersihkan karakter OCR yang salah
    3. Pertahankan prefix "J"
    """
    if not raw:
        return None

    cleaned = raw.strip().upper().replace(" ", "")
    # Hapus prefix "NIM" atau "NIM." jika ada
    cleaned = re.sub(r"^N[I1]M\.?\s*", "", cleaned)
    logger.debug(f"  NIM cleaning: '{raw}' → after strip: '{cleaned}'")

    # ── Strategi 1: Deteksi NIM format IPB dengan prefix J ──
    # Pattern: J diikuti minimal 6 digit/karakter
    j_match = re.search(r"([JI1])([\dA-Za-z|.,]{6,12})", cleaned)

    if j_match:
        prefix = "J"  # Lock prefix sebagai J
        raw_suffix = j_match.group(2)

        # Cek apakah bagian kedua ada "3B" (format J3B)
        if len(raw_suffix) >= 2 and raw_suffix[0] == "3" and raw_suffix[1] in "B8":
            prefix = "J3B"
            raw_suffix = raw_suffix[2:]

        # Bersihkan suffix — konversi OCR errors ke digit [M2]
        fixed_suffix = ""
        for ch in raw_suffix:
            if ch.isdigit():
                fixed_suffix += ch
            elif ch in _STRIP_CHARS:
                continue  # skip whitespace/punctuation
            elif ch in _OCR_TO_DIGIT:
                fixed_suffix += _OCR_TO_DIGIT[ch]

        nim = prefix + fixed_suffix
        logger.info(f"  NIM cleaned (IPB format): '{raw}' → '{nim}'")
        return nim

    # ── Strategi 2: Fallback — ambil semua digit ──
    digits_only = ""
    for ch in cleaned:
        if ch.isdigit():
            digits_only += ch
        elif ch in _STRIP_CHARS:
            continue
        elif ch in _OCR_TO_DIGIT:
            digits_only += _OCR_TO_DIGIT[ch]

    if digits_only:
        logger.info(f"  NIM cleaned (digits only): '{raw}' → '{digits_only}'")
        return digits_only

    logger.warning(f"  ⚠️ NIM cleaning failed, no digits found in: '{raw}'")
    return None


# ────────────────────────────────────────────────────────
# Name Cleaning
# ────────────────────────────────────────────────────────
def clean_name(raw: Optional[str]) -> Optional[str]:
    """
    Bersihkan raw OCR output dari Nama:
    1. Hapus trailing NIM pattern (dari bbox overlap)
    2. Hapus karakter non-alfabet
    3. Fix common OCR first-letter errors
    4. Title case
    """
    if not raw:
        return None

    # Strip trailing NIM/VIM + digits (\\s+ required to avoid matching M in MUHAMMAD)
    cleaned = re.sub(r"\s+(N?IM|V/?I?M|NIM)[\s.,/]*[JjIi]?\d*.*$", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"[\s.,/]+[JjIi]?\d{4,}.*$", "", cleaned)
    cleaned = re.sub(r"[\s.,/]*[A-Z]?\.[JjIi]\d+.*$", "", cleaned)

    # Hapus karakter aneh, sisakan huruf + spasi + . + '
    cleaned = re.sub(r"[^a-zA-Z\s.']", "", cleaned)
    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Strip trailing single char (artifact: 'Zaldiputra V' → 'Zaldiputra')
    cleaned = re.sub(r"\s+[A-Z]$", "", cleaned)
    # Title case
    cleaned = cleaned.title()

    if len(cleaned) < 2:
        logger.warning(f"  ⚠️ Name too short after cleaning: '{raw}' → '{cleaned}'")
        return None

    logger.info(f"  Name cleaned: '{raw}' → '{cleaned}'")
    return cleaned


# ────────────────────────────────────────────────────────
# Double Validation
# ────────────────────────────────────────────────────────
def validate_nim_match(
    nim_qr: Optional[str],
    nim_ocr: Optional[str],
) -> Tuple[bool, str]:
    """
    Double validation: cocokkan NIM dari QR code dengan NIM dari OCR teks.

    Returns:
        (is_valid, detail_message)
    """
    if not nim_qr and not nim_ocr:
        return False, "❌ Both QR NIM and OCR NIM are empty."

    if not nim_qr:
        return False, f"⚠️ QR NIM is empty. OCR NIM: '{nim_ocr}'"

    if not nim_ocr:
        return False, f"⚠️ OCR NIM is empty. QR NIM: '{nim_qr}'"

    # Normalize untuk perbandingan
    qr_norm = nim_qr.strip().upper()
    ocr_norm = nim_ocr.strip().upper()

    if qr_norm == ocr_norm:
        return True, f"✅ NIM MATCH: '{qr_norm}'"

    # Cek similarity — mungkin beda 1-2 karakter (OCR noise)
    # Hitung karakter yang sama
    if len(qr_norm) == len(ocr_norm):
        diff_count = sum(1 for a, b in zip(qr_norm, ocr_norm) if a != b)
        if diff_count <= 2:
            return False, (
                f"⚠️ NIM PARTIAL MATCH ({diff_count} char diff): "
                f"QR='{qr_norm}' vs OCR='{ocr_norm}'"
            )

    return False, f"❌ NIM MISMATCH: QR='{qr_norm}' vs OCR='{ocr_norm}'"


def extract_nim_from_qr_data(qr_data: Optional[str]) -> Optional[str]:
    """
    QR code di KTM IPB mungkin berisi:
    - NIM langsung: "J0403231061"
    - NIM format lama: "J3B121015"
    - URL: "https://example.com/student/J0403231061"

    Fungsi ini mengekstrak NIM dari berbagai format.
    """
    if not qr_data:
        return None

    # Pattern 1: J diikuti 7-12 digit (format baru, misal J0403231061)
    match = re.search(r"[Jj]\d{7,12}", qr_data)
    if match:
        return match.group().upper()

    # Pattern 2: J3B format (J + digit + B + 6 digit)
    match = re.search(r"[Jj]\d[Bb]\d{6}", qr_data)
    if match:
        return match.group().upper()

    # Pattern 3: angka panjang (>= 7 digit) yang mungkin NIM tanpa prefix
    match = re.search(r"\d{7,}", qr_data)
    if match:
        return match.group()

    # Jika QR data pendek dan mungkin NIM itu sendiri
    cleaned = qr_data.strip()
    if 6 <= len(cleaned) <= 15:
        return cleaned

    return None
