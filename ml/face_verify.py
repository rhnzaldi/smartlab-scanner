"""
Smart-Lab SV IPB — Face Verification Module (InsightFace ArcFace)
Face enrollment + verification via webcam.

Engine: InsightFace buffalo_l (ArcFace w600k_r50)
- 512-D face embedding
- 99.5%+ akurasi untuk wajah Asia
- ~50-80ms per frame via ONNX Runtime

Flow:
  1. ENROLLMENT: Capture wajah via webcam → 512-D embedding → simpan di DB
  2. VERIFICATION: Load embedding dari DB → compare wajah live

Privacy: Hanya menyimpan 512 angka (embedding), BUKAN foto wajah.
"""

import logging
from typing import Dict

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────
# Lazy-load InsightFace
# ────────────────────────────────────────────────────────
_face_app = None
_face_app_available = None


def _get_face_app():
    """Lazy-load InsightFace FaceAnalysis. Hanya load detection + recognition."""
    global _face_app, _face_app_available
    if _face_app_available is False:
        return None
    if _face_app is None:
        try:
            from insightface.app import FaceAnalysis
            # Hanya load detection + recognition (skip landmark + genderage)
            # Hemat ~100MB RAM, tidak pengaruh akurasi
            _face_app = FaceAnalysis(
                name='buffalo_l',
                providers=['CPUExecutionProvider'],
                allowed_modules=['detection', 'recognition'],
            )
            # det_size lebih besar = deteksi wajah lebih akurat
            _face_app.prepare(ctx_id=-1, det_size=(640, 640))
            _face_app_available = True
            logger.info("✅ InsightFace ArcFace loaded (buffalo_l, det_size=640)")
        except Exception as e:
            logger.error(f"❌ InsightFace gagal dimuat: {e}")
            _face_app_available = False
            return None
    return _face_app


# ────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────
FACE_MATCH_THRESHOLD = 0.5   # ≥50% similarity = cocok, <50% = DITOLAK


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Hitung cosine similarity antara dua embedding."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


class FaceVerifier:
    """
    Face enrollment + verification menggunakan InsightFace ArcFace.

    Enrollment:  capture wajah live → 512-D embedding → simpan di DB
    Verification: load embedding dari DB → compare wajah live
    """

    def __init__(self, threshold: float = FACE_MATCH_THRESHOLD):
        self.threshold = threshold
        self._ref_embedding = None    # 512-D numpy array (from DB)
        self._has_reference = False

    def set_reference_from_encoding(self, embedding: np.ndarray) -> bool:
        """Load embedding yang sudah di-save dari DB."""
        if embedding is None or len(embedding) < 64:
            return False
        self._ref_embedding = embedding.astype(np.float32)
        self._has_reference = True
        logger.info(f"✅ Face reference loaded from DB ({len(embedding)}-D)")
        return True

    def enroll(self, live_frame: np.ndarray) -> Dict:
        """
        ENROLLMENT: Capture wajah dari webcam dan hasilkan embedding.
        Returns embedding (512-D numpy array) jika berhasil.
        """
        result = {
            "success": False,
            "encoding": None,
            "face_detected": False,
            "face_bbox": None,
            "message": "",
        }

        app = _get_face_app()
        if app is None:
            result["message"] = "InsightFace not available"
            return result

        rgb = cv2.cvtColor(live_frame, cv2.COLOR_BGR2RGB)
        faces = app.get(rgb)

        if not faces:
            result["message"] = "Wajah tidak terdeteksi — hadap ke kamera"
            return result

        # Get largest face (closest to camera)
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        x1, y1, x2, y2 = [int(v) for v in largest.bbox]

        result["face_detected"] = True
        result["face_bbox"] = (x1, y1, x2 - x1, y2 - y1)

        if largest.embedding is None:
            result["message"] = "Gagal encode wajah"
            return result

        result["success"] = True
        result["encoding"] = largest.embedding  # 512-D float32
        result["message"] = "✅ Wajah berhasil di-capture"
        logger.info(f"✅ Face enrollment: {len(largest.embedding)}-D ArcFace embedding")
        return result

    def verify(self, live_frame: np.ndarray) -> Dict:
        """
        VERIFICATION: Compare wajah live vs embedding referensi dari DB.
        Proses setiap frame untuk akurasi real-time terbaik.
        """
        result = {
            "verified": False,
            "similarity": 0.0,
            "face_detected": False,
            "message": "",
            "face_bbox": None,
        }

        app = _get_face_app()
        if app is None or not self._has_reference:
            result["message"] = "Not available" if app is None else "No reference"
            return result

        rgb = cv2.cvtColor(live_frame, cv2.COLOR_BGR2RGB)
        faces = app.get(rgb)

        if not faces:
            result["message"] = "Wajah tidak terdeteksi — hadap ke kamera"
            return result

        # Largest face
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        x1, y1, x2, y2 = [int(v) for v in largest.bbox]

        result["face_detected"] = True
        result["face_bbox"] = (x1, y1, x2 - x1, y2 - y1)

        if largest.embedding is None:
            result["message"] = "Error encoding wajah"
            return result

        # Cosine similarity
        similarity = _cosine_similarity(self._ref_embedding, largest.embedding)
        result["similarity"] = round(max(0.0, similarity), 2)

        if similarity >= self.threshold:
            result["verified"] = True
            result["message"] = f"✅ Wajah cocok ({similarity:.0%})"
            logger.info(f"Face MATCH: sim={similarity:.3f} ≥ {self.threshold}")
        else:
            result["message"] = f"Wajah tidak cocok ({similarity:.0%})"
            logger.debug(f"Face REJECT: sim={similarity:.3f} < {self.threshold}")

        return result

    @property
    def has_reference(self) -> bool:
        return self._has_reference

    def clear_reference(self):
        """Reset referensi."""
        self._ref_embedding = None
        self._has_reference = False
