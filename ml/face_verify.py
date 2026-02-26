"""
Smart-Lab SV IPB — Face Verification Module
Face enrollment + verification via webcam menggunakan dlib 128-D encoding.

Flow:
  1. ENROLLMENT (pertama kali):
     Scan KTM → DB match → Capture wajah via webcam → Simpan encoding di DB
  2. VERIFICATION (selanjutnya):
     Scan KTM → DB match → Load encoding dari DB → Compare wajah live

Privacy: Hanya menyimpan 128 angka (encoding), BUKAN foto wajah.
"""

import logging
from typing import Optional, Dict

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────
# Lazy-load face_recognition
# ────────────────────────────────────────────────────────
_face_recognition = None
_fr_available = None


def _get_face_recognition():
    """Lazy-load face_recognition module."""
    global _face_recognition, _fr_available
    if _fr_available is False:
        return None
    if _face_recognition is None:
        try:
            import face_recognition
            _face_recognition = face_recognition
            _fr_available = True
            logger.info("✅ face_recognition (dlib) loaded")
        except ImportError:
            logger.error(
                "❌ face_recognition not installed.\n"
                "   Install: conda install -c conda-forge dlib && pip install face_recognition"
            )
            _fr_available = False
            return None
    return _face_recognition


# ────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────
FACE_MATCH_TOLERANCE = 0.45      # euclidean distance (lower = stricter)
FACE_DETECTION_MODEL = "hog"     # "hog" (CPU fast) or "cnn" (GPU accurate)


class FaceVerifier:
    """
    Face enrollment + verification.

    Enrollment:  capture wajah live → encoding → simpan di DB
    Verification: load encoding dari DB → compare wajah live
    """

    def __init__(self, tolerance: float = FACE_MATCH_TOLERANCE):
        self.tolerance = tolerance
        self._ref_encoding = None   # 128-D numpy array (from DB)
        self._has_reference = False

    def set_reference_from_encoding(self, encoding: np.ndarray) -> bool:
        """Load encoding yang sudah di-save dari DB."""
        if encoding is None or len(encoding) != 128:
            return False
        self._ref_encoding = encoding
        self._has_reference = True
        logger.info("✅ Face reference loaded from DB (128-D)")
        return True

    def enroll(self, live_frame: np.ndarray) -> Dict:
        """
        ENROLLMENT: Capture wajah dari webcam dan hasilkan encoding.
        Returns encoding (128-D numpy array) jika berhasil.

        Caller harus save encoding ke DB via save_face_encoding().
        """
        result = {
            "success": False,
            "encoding": None,
            "face_detected": False,
            "face_bbox": None,
            "message": "",
        }

        fr = _get_face_recognition()
        if fr is None:
            result["message"] = "face_recognition not available"
            return result

        rgb = cv2.cvtColor(live_frame, cv2.COLOR_BGR2RGB)
        face_locations = fr.face_locations(rgb, model=FACE_DETECTION_MODEL)

        if not face_locations:
            result["message"] = "Wajah tidak terdeteksi — hadap ke kamera"
            return result

        # Get largest face
        largest_face = max(face_locations, key=lambda l: (l[2] - l[0]) * (l[1] - l[3]))
        top, right, bottom, left = largest_face

        result["face_detected"] = True
        result["face_bbox"] = (left, top, right - left, bottom - top)

        # Encode with high quality (num_jitters=5 for enrollment)
        encodings = fr.face_encodings(rgb, known_face_locations=[largest_face],
                                      num_jitters=5)
        if not encodings:
            result["message"] = "Gagal encode wajah"
            return result

        result["success"] = True
        result["encoding"] = encodings[0]
        result["message"] = "✅ Wajah berhasil di-capture"
        logger.info("✅ Face enrollment: encoding captured (128-D, jitters=5)")
        return result

    def verify(self, live_frame: np.ndarray) -> Dict:
        """
        VERIFICATION: Compare wajah live vs encoding referensi dari DB.
        """
        result = {
            "verified": False,
            "similarity": 0.0,
            "distance": 1.0,
            "face_detected": False,
            "message": "",
            "face_bbox": None,
        }

        fr = _get_face_recognition()
        if fr is None or not self._has_reference:
            result["message"] = "Not available" if fr is None else "No reference"
            return result

        rgb = cv2.cvtColor(live_frame, cv2.COLOR_BGR2RGB)
        face_locations = fr.face_locations(rgb, model=FACE_DETECTION_MODEL)

        if not face_locations:
            result["message"] = "Wajah tidak terdeteksi — hadap ke kamera"
            return result

        # Largest face
        largest_face = max(face_locations, key=lambda l: (l[2] - l[0]) * (l[1] - l[3]))
        top, right, bottom, left = largest_face

        result["face_detected"] = True
        result["face_bbox"] = (left, top, right - left, bottom - top)

        # Encode live face (jitters=1 for speed)
        live_encodings = fr.face_encodings(rgb, known_face_locations=[largest_face],
                                           num_jitters=1)
        if not live_encodings:
            result["message"] = "Error encoding wajah"
            return result

        # Compare
        distance = float(fr.face_distance([self._ref_encoding], live_encodings[0])[0])
        similarity = max(0.0, min(1.0, 1.0 - distance))

        result["distance"] = round(distance, 3)
        result["similarity"] = round(similarity, 2)

        if distance <= self.tolerance:
            result["verified"] = True
            result["message"] = f"✅ Wajah cocok ({similarity:.0%})"
            logger.info(f"Face MATCH: dist={distance:.3f} ≤ {self.tolerance}")
        else:
            result["message"] = f"Wajah tidak cocok ({similarity:.0%})"
            logger.debug(f"Face REJECT: dist={distance:.3f} > {self.tolerance}")

        return result

    @property
    def has_reference(self) -> bool:
        return self._has_reference

    def clear_reference(self):
        """Reset referensi."""
        self._ref_encoding = None
        self._has_reference = False
