"""
Smart-Lab SV IPB — Face Verification Module
Bandingkan wajah live dari webcam dengan foto di KTM.

Menggunakan face_recognition (dlib) untuk 128-D face encoding:
1. dlib HOG/CNN untuk deteksi wajah
2. 128-dimensional face encoding untuk matching
3. Euclidean distance untuk similarity score

Usage:
    from ml.face_verify import FaceVerifier
    verifier = FaceVerifier()

    # Simpan referensi dari KTM crop
    verifier.set_reference(ktm_face_crop)

    # Verify wajah live
    result = verifier.verify(live_frame)
    print(result["verified"], result["similarity"])
"""

import logging
from typing import Optional, Dict, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────
# Lazy-load face_recognition (heavy import)
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
FACE_MATCH_TOLERANCE = 0.5      # euclidean distance threshold (lower = stricter)
FACE_DETECTION_MODEL = "hog"    # "hog" (CPU, fast) or "cnn" (GPU, accurate)


class FaceVerifier:
    """
    Verify wajah live vs referensi dari KTM.

    Flow:
    1. set_reference(ktm_face_crop) — encode face dari foto KTM
    2. verify(live_frame) — detect face live, compare encoding
    """

    def __init__(self, tolerance: float = FACE_MATCH_TOLERANCE):
        self.tolerance = tolerance
        self._ref_encoding = None   # 128-D numpy array
        self._ref_image = None
        self._has_reference = False

    def set_reference(self, face_crop: np.ndarray) -> bool:
        """
        Set foto referensi dari KTM face_photo crop.
        Returns True jika berhasil mendeteksi dan encode wajah.
        """
        fr = _get_face_recognition()
        if fr is None:
            return False

        if face_crop is None or face_crop.size == 0:
            logger.warning("⚠️ Face crop kosong")
            return False

        # face_recognition expects RGB (OpenCV gives BGR)
        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        # Upscale small crops for better detection
        h, w = rgb.shape[:2]
        if h < 100 or w < 80:
            scale = max(100 / h, 80 / w)
            rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            logger.debug(f"  Upscaled face crop: {w}x{h} → {rgb.shape[1]}x{rgb.shape[0]}")

        # Get face encodings
        encodings = fr.face_encodings(rgb)

        if not encodings:
            # KTM crop mungkin sudah berupa wajah (YOLO crop ketat)
            # Coba tanpa face detection — encode seluruh gambar
            encodings = fr.face_encodings(rgb, known_face_locations=[(0, rgb.shape[1], rgb.shape[0], 0)])

        if not encodings:
            logger.warning("⚠️ Tidak bisa encode wajah dari KTM crop")
            return False

        self._ref_encoding = encodings[0]
        self._ref_image = face_crop.copy()
        self._has_reference = True
        logger.info("✅ Face reference encoding set dari KTM (128-D)")
        return self._has_reference

    def verify(self, live_frame: np.ndarray) -> Dict:
        """
        Verify wajah di live frame vs referensi.

        Returns:
            {
                "verified": bool,
                "similarity": float (0-1, higher = more similar),
                "distance": float (euclidean, lower = more similar),
                "face_detected": bool,
                "message": str,
                "face_bbox": (x, y, w, h) | None,
            }
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
            result["message"] = "Face recognition not available" if fr is None else "No reference set"
            return result

        # Convert BGR → RGB
        rgb = cv2.cvtColor(live_frame, cv2.COLOR_BGR2RGB)

        # Detect faces (HOG = fast CPU, CNN = accurate GPU)
        face_locations = fr.face_locations(rgb, model=FACE_DETECTION_MODEL)

        if not face_locations:
            result["message"] = "Wajah tidak terdeteksi — hadap ke kamera"
            return result

        # Get largest face (closest to camera)
        largest_face = max(face_locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))
        top, right, bottom, left = largest_face

        result["face_detected"] = True
        result["face_bbox"] = (left, top, right - left, bottom - top)  # (x, y, w, h)

        # Encode live face
        live_encodings = fr.face_encodings(rgb, known_face_locations=[largest_face])
        if not live_encodings:
            result["message"] = "Error encoding wajah live"
            return result

        # Compare: euclidean distance
        distance = fr.face_distance([self._ref_encoding], live_encodings[0])[0]
        similarity = max(0.0, 1.0 - distance)  # convert distance to similarity (0-1)

        result["distance"] = round(float(distance), 3)
        result["similarity"] = round(float(similarity), 2)

        if distance <= self.tolerance:
            result["verified"] = True
            result["message"] = f"✅ Wajah cocok ({similarity:.0%})"
            logger.info(f"Face verified: distance={distance:.3f}, similarity={similarity:.0%}")
        else:
            result["message"] = f"❌ Wajah tidak cocok ({similarity:.0%})"
            logger.debug(f"Face mismatch: distance={distance:.3f}, similarity={similarity:.0%}")

        return result

    @property
    def has_reference(self) -> bool:
        return self._has_reference

    def clear_reference(self):
        """Reset referensi."""
        self._ref_encoding = None
        self._ref_image = None
        self._has_reference = False
