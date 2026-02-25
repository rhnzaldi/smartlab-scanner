"""
Smart-Lab SV IPB — Face Verification Module
Bandingkan wajah live dari webcam dengan foto di KTM.

Menggunakan pure OpenCV (tanpa library tambahan):
1. Haar Cascade untuk deteksi wajah
2. Histogram comparison untuk matching

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
from typing import Optional, Dict

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Load Haar Cascade face detector (bundled with OpenCV) [R5]
_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_cascade_path)

if _face_cascade.empty():
    logger.error(
        f"❌ Haar Cascade gagal dimuat dari: {_cascade_path}\n"
        f"   Face verification tidak akan berfungsi."
    )

# Threshold for face match
FACE_MATCH_THRESHOLD = 0.55  # similarity 0-1, > threshold = match


class FaceVerifier:
    """
    Verify wajah live vs referensi dari KTM.

    Flow:
    1. set_reference(ktm_face_crop) — simpan face dari KTM
    2. verify(live_frame) — capture wajah live, compare
    """

    def __init__(self, threshold: float = FACE_MATCH_THRESHOLD):
        self.threshold = threshold
        self._ref_hist = None  # histogram referensi
        self._ref_image = None
        self._has_reference = False

    def set_reference(self, face_crop: np.ndarray) -> bool:
        """
        Set foto referensi dari KTM face_photo crop.
        Returns True jika berhasil mendeteksi wajah.
        """
        if face_crop is None or face_crop.size == 0:
            logger.warning("⚠️ Face crop kosong")
            return False

        # Detect face in KTM crop
        face_roi = self._detect_and_crop_face(face_crop)
        if face_roi is None:
            # KTM crop mungkin sudah berupa wajah (karena YOLO crop)
            # Gunakan langsung
            face_roi = face_crop
            logger.debug("  Using entire KTM crop as face reference")

        # Compute histogram
        self._ref_hist = self._compute_histogram(face_roi)
        self._ref_image = face_roi.copy()
        self._has_reference = self._ref_hist is not None

        if self._has_reference:
            logger.info("✅ Face reference set from KTM")
        else:
            logger.warning("⚠️ Failed to compute face reference histogram")

        return self._has_reference

    def verify(self, live_frame: np.ndarray) -> Dict:
        """
        Verify wajah di live frame vs referensi.

        Returns:
            {
                "verified": bool,
                "similarity": float (0-1),
                "face_detected": bool,
                "message": str,
                "face_bbox": (x, y, w, h) | None,
            }
        """
        result = {
            "verified": False,
            "similarity": 0.0,
            "face_detected": False,
            "message": "",
            "face_bbox": None,
        }

        if not self._has_reference:
            result["message"] = "No face reference set"
            return result

        # Detect face in live frame
        gray = cv2.cvtColor(live_frame, cv2.COLOR_BGR2GRAY)
        faces = _face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )

        if len(faces) == 0:
            result["message"] = "Wajah tidak terdeteksi — hadap ke kamera"
            return result

        # Get largest face (closest to camera)
        face_areas = [w * h for (x, y, w, h) in faces]
        largest_idx = face_areas.index(max(face_areas))
        x, y, w, h = faces[largest_idx]

        result["face_detected"] = True
        result["face_bbox"] = (int(x), int(y), int(w), int(h))

        # Crop face from live frame
        live_face = live_frame[y:y+h, x:x+w]

        # Compare histograms
        live_hist = self._compute_histogram(live_face)
        if live_hist is None:
            result["message"] = "Face processing error"
            return result

        # Correlation-based similarity (0 to 1)
        similarity = cv2.compareHist(
            self._ref_hist, live_hist, cv2.HISTCMP_CORREL
        )
        # Normalize to 0-1 range
        similarity = max(0.0, min(1.0, similarity))

        result["similarity"] = round(similarity, 2)

        if similarity >= self.threshold:
            result["verified"] = True
            result["message"] = f"✅ Wajah cocok ({similarity:.0%})"
            logger.info(f"Face verified: {similarity:.0%}")
        else:
            result["message"] = f"❌ Wajah tidak cocok ({similarity:.0%})"
            logger.warning(f"Face mismatch: {similarity:.0%}")

        return result

    @property
    def has_reference(self) -> bool:
        return self._has_reference

    def clear_reference(self):
        """Reset referensi."""
        self._ref_hist = None
        self._ref_image = None
        self._has_reference = False

    @staticmethod
    def _detect_and_crop_face(img: np.ndarray) -> Optional[np.ndarray]:
        """Detect and crop the largest face from image."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        faces = _face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
        )
        if len(faces) == 0:
            return None
        # Ambil wajah terbesar
        face_areas = [w * h for (x, y, w, h) in faces]
        largest_idx = face_areas.index(max(face_areas))
        x, y, w, h = faces[largest_idx]
        return img[y:y+h, x:x+w]

    @staticmethod
    def _compute_histogram(face_img: np.ndarray) -> Optional[np.ndarray]:
        """Compute normalized color histogram for face comparison."""
        try:
            # Resize to standard size for fair comparison
            face_resized = cv2.resize(face_img, (128, 128))

            # Convert to HSV (better for face comparison than BGR)
            if len(face_resized.shape) == 3:
                hsv = cv2.cvtColor(face_resized, cv2.COLOR_BGR2HSV)
            else:
                hsv = face_resized

            # Compute histogram (H and S channels)
            hist = cv2.calcHist(
                [hsv], [0, 1], None,
                [50, 60], [0, 180, 0, 256]
            )
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            return hist
        except Exception as e:
            logger.error(f"Histogram error: {e}")
            return None
