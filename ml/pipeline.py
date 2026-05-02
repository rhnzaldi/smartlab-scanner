"""
Smart-Lab SV IPB — KTM Pipeline
Orchestrator utama: YOLO detect → crop → preprocess → extract → validate.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .preprocessor import preprocess_for_ocr, preprocess_for_qr
from .extractor import extract_qr, extract_text_ocr
from .validator import (
    clean_nim,
    clean_name,
    validate_nim_match,
    extract_nim_from_qr_data,
)

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────
# Data Classes
# ────────────────────────────────────────────────────────

@dataclass
class Detection:
    """Satu objek yang terdeteksi YOLO."""
    label: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    crop: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class ScanResult:
    """Hasil akhir scanning satu frame KTM."""
    success: bool = False
    status: str = "no_detection"

    # Extracted data
    nim_qr: Optional[str] = None
    nim_ocr: Optional[str] = None
    nim_final: Optional[str] = None
    nama: Optional[str] = None

    # Validation
    nim_match: bool = False
    validation_detail: str = ""

    # Detections metadata
    detections_found: List[str] = field(default_factory=list)
    confidences: Dict[str, float] = field(default_factory=dict)

    # Timing
    inference_time_ms: float = 0.0
    total_time_ms: float = 0.0

    # Bounding boxes for drawing
    bboxes: Dict[str, List[int]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize ke dict (untuk JSON response)."""
        return {
            "success": self.success,
            "status": self.status,
            "nim_qr": self.nim_qr,
            "nim_ocr": self.nim_ocr,
            "nim_final": self.nim_final,
            "nama": self.nama,
            "nim_match": self.nim_match,
            "validation_detail": self.validation_detail,
            "detections_found": self.detections_found,
            "confidences": self.confidences,
            "bboxes": self.bboxes,
            "inference_time_ms": round(self.inference_time_ms, 1),
            "total_time_ms": round(self.total_time_ms, 1),
        }


# ────────────────────────────────────────────────────────
# YOLO Label Mapping
# ────────────────────────────────────────────────────────
# Class names yang diharapkan dari model YOLO
EXPECTED_LABELS = {"qr_code", "text_nim", "text_nama", "face_photo"}


# ────────────────────────────────────────────────────────
# KTM Pipeline
# ────────────────────────────────────────────────────────
class KTMPipeline:
    """
    Pipeline utama untuk memproses frame webcam dan mengekstrak data KTM.

    Usage:
        pipeline = KTMPipeline(model_path="models/best.pt")
        result = pipeline.process_frame(frame)
        print(result.to_dict())
    """

    def __init__(
        self,
        model_path: str = "models/best.pt",
        confidence_threshold: float = 0.50,   # dinaikkan dari 0.35 → model sudah 97%
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None
        self._model_loaded = False
        self._class_names: Dict[int, str] = {}

        self._load_model()

    def _load_model(self):
        """Load YOLOv8 model. Graceful jika file belum ada."""
        abs_path = os.path.abspath(self.model_path)

        if not os.path.exists(abs_path):
            logger.error(
                f"❌ Model file not found: {abs_path}\n"
                f"   Please copy your trained best.pt to: {abs_path}\n"
                f"   Pipeline will return empty results until model is available."
            )
            self._model_loaded = False
            return

        try:
            from ultralytics import YOLO
            logger.info(f"⏳ Loading YOLO model from {abs_path}...")
            self.model = YOLO(abs_path)
            self._class_names = self.model.names  # {0: 'qr_code', 1: 'nim_teks', ...}
            self._model_loaded = True
            logger.info(
                f"✅ YOLO model loaded. Classes: {self._class_names}"
            )
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO model: {e}")
            self._model_loaded = False

    def is_ready(self) -> bool:
        """Check apakah pipeline siap dipakai."""
        return self._model_loaded and self.model is not None

    # ────────────────── YOLO Detection ──────────────────

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Jalankan YOLO inference pada frame.
        Returns list of Detection objects.

        Error handling ketat karena model dilatih dengan hanya 85 gambar.
        """
        if not self.is_ready():
            logger.warning("⚠️ Pipeline not ready (model not loaded).")
            return []

        try:
            t_start = time.perf_counter()

            results = self.model.predict(
                source=frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )

            t_elapsed = (time.perf_counter() - t_start) * 1000

            if not results or len(results) == 0:
                logger.debug(f"YOLO: no results ({t_elapsed:.0f}ms)")
                return []

            detections: List[Detection] = []
            result = results[0]

            if result.boxes is None or len(result.boxes) == 0:
                logger.debug(f"YOLO: no boxes ({t_elapsed:.0f}ms)")
                return []

            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                label = self._class_names.get(cls_id, f"class_{cls_id}")

                detections.append(Detection(
                    label=label,
                    confidence=conf,
                    bbox=[x1, y1, x2, y2],
                ))

            logger.debug(
                f"YOLO: {len(detections)} detections in {t_elapsed:.0f}ms "
                f"[{', '.join(d.label for d in detections)}]"
            )
            return detections

        except Exception as e:
            logger.error(f"❌ YOLO detection error: {e}")
            return []

    # ────────────────── Crop ──────────────────

    @staticmethod
    def crop_detections(
        frame: np.ndarray,
        detections: List[Detection],
    ) -> Dict[str, Detection]:
        """
        Crop bounding box regions dari frame.
        Ditambahkan Auto-Padding margin putih agar OCR tidak memotong huruf awal/akhir.
        Returns dict mapping label → Detection (with crop filled).
        Jika ada duplikat label, ambil yang confidence tertinggi.
        """
        h, w = frame.shape[:2]
        crops: Dict[str, Detection] = {}

        for det in detections:
            # Padding margins (dalam pixel)
            # Teks butuh padding X lebih banyak agar kerning/huruf tepi masuk semua
            pad_x_left, pad_x_right, pad_y = 0, 0, 0
            
            if det.label == "text_nim":
                pad_x_left, pad_x_right, pad_y = 15, 30, 8
            elif det.label == "text_nama":
                pad_x_left, pad_x_right, pad_y = 15, 45, 10
            elif det.label == "qr_code":
                pad_x_left, pad_x_right, pad_y = 10, 10, 10

            # Set kotak dengan padding (tidak boleh melebihi batas ukuran/resolusi gambar frame)
            x1 = max(0, det.bbox[0] - pad_x_left)
            y1 = max(0, det.bbox[1] - pad_y)
            x2 = min(w, det.bbox[2] + pad_x_right)
            y2 = min(h, det.bbox[3] + pad_y)

            if x2 - x1 < 5 or y2 - y1 < 5:
                logger.warning(
                    f"⚠️ Crop too small for '{det.label}': "
                    f"{x2-x1}x{y2-y1}px, skipping."
                )
                continue

            det.crop = frame[y1:y2, x1:x2].copy()

            # Keep highest confidence detection per label
            if det.label not in crops or det.confidence > crops[det.label].confidence:
                crops[det.label] = det

        return crops

    # ────────────────── Full Pipeline ──────────────────

    def process_frame(self, frame: np.ndarray) -> ScanResult:
        """
        Proses satu frame melalui pipeline lengkap:
        1. YOLO detect
        2. Crop
        3. Preprocess
        4. Extract (QR + OCR)
        5. Clean & Validate

        Returns ScanResult with all extracted data.
        """
        result = ScanResult()
        t_total_start = time.perf_counter()

        # ── Guard: model not ready ──
        if not self.is_ready():
            result.status = "model_not_loaded"
            return result

        # ── Guard: invalid frame ──
        if frame is None or frame.size == 0:
            result.status = "invalid_frame"
            return result

        # ── Step 1: YOLO Detection ──
        t_infer_start = time.perf_counter()
        detections = self.detect(frame)
        result.inference_time_ms = (time.perf_counter() - t_infer_start) * 1000

        if not detections:
            result.status = "no_detection"
            result.total_time_ms = (time.perf_counter() - t_total_start) * 1000
            return result

        # ── Step 2: Crop ──
        crops = self.crop_detections(frame, detections)
        result.detections_found = list(crops.keys())
        result.confidences = {k: round(v.confidence, 3) for k, v in crops.items()}
        result.bboxes = {k: v.bbox for k, v in crops.items()}

        # ── Step 3 & 4: Extract QR Code ──
        if "qr_code" in crops and crops["qr_code"].crop is not None:
            qr_crop = crops["qr_code"].crop
            # preprocess_for_qr hanya upscale + grayscale (tidak binary)
            # Strategi binerisasi / CLAHE dilakukan di dalam extract_qr() sendiri
            qr_preprocessed = preprocess_for_qr(qr_crop)   # upscale ke ≥200px
            qr_raw = extract_qr(qr_preprocessed)            # 9 strategi decode

            # Fallback: coba langsung dari raw BGR crop (siapa tahu lebih baik)
            if not qr_raw:
                logger.debug("QR: preprocessed failed, retrying with raw color crop...")
                qr_raw = extract_qr(qr_crop)

            if qr_raw:
                result.nim_qr = extract_nim_from_qr_data(qr_raw)
                if not result.nim_qr:
                    result.nim_qr = qr_raw  # simpan raw jika tidak match pattern

        # ── Step 3 & 4: Extract NIM Text ──
        if "text_nim" in crops and crops["text_nim"].crop is not None:
            nim_crop = crops["text_nim"].crop
            nim_preprocessed = preprocess_for_ocr(nim_crop)
            nim_raw = extract_text_ocr(nim_preprocessed)
            if not nim_raw:
                nim_raw = extract_text_ocr(nim_crop)
            result.nim_ocr = clean_nim(nim_raw)

        # ── Step 3 & 4: Extract Nama Text ──
        if "text_nama" in crops and crops["text_nama"].crop is not None:
            nama_crop = crops["text_nama"].crop
            nama_preprocessed = preprocess_for_ocr(nama_crop)
            nama_raw = extract_text_ocr(nama_preprocessed)
            if not nama_raw:
                nama_raw = extract_text_ocr(nama_crop)
            result.nama = clean_name(nama_raw)

        # ── Step 5: Double Validation (4 Indikator) ──
        # Keempat indikator yang WAJIB ada untuk validasi penuh:
        #   1. qr_code   → nim_qr
        #   2. text_nim   → nim_ocr
        #   3. text_nama  → nama
        #   4. face_photo → terdeteksi YOLO (bukti objek = KTM asli)
        nim_match, detail = validate_nim_match(result.nim_qr, result.nim_ocr)
        result.nim_match = nim_match
        result.validation_detail = detail

        # Determine final NIM (tetap simpan untuk logging/debug)
        if nim_match:
            result.nim_final = result.nim_qr
        elif result.nim_qr:
            result.nim_final = result.nim_qr
        elif result.nim_ocr:
            result.nim_final = result.nim_ocr

        # ── Determine overall success ──
        # Cek kehadiran keempat indikator
        has_qr = result.nim_qr is not None
        has_nim_ocr = result.nim_ocr is not None
        has_nama = result.nama is not None
        has_face_photo = "face_photo" in result.detections_found

        # SUKSES hanya jika:
        #   - QR code berhasil di-decode (has_qr)
        #   - NIM OCR berhasil di-baca (has_nim_ocr)
        #   - Nama berhasil di-baca (has_nama)
        #   - Face photo terdeteksi YOLO (has_face_photo)
        #   - NIM dari QR == NIM dari OCR (nim_match)
        all_four_present = has_qr and has_nim_ocr and has_nama and has_face_photo
        result.success = all_four_present and nim_match

        if result.success:
            result.status = "validated"
        elif all_four_present and not nim_match:
            result.status = "nim_mismatch"
        elif not has_qr and (has_nim_ocr or has_nama):
            result.status = "qr_missing"
        elif not has_face_photo and (has_nim_ocr or has_nama):
            result.status = "face_photo_missing"
        elif result.nim_final or has_nama:
            result.status = "incomplete"
        else:
            result.status = "extraction_failed"

        # Log detail untuk setiap indikator
        indicators = (
            f"QR={'✅' if has_qr else '❌'} "
            f"NIM_OCR={'✅' if has_nim_ocr else '❌'} "
            f"Nama={'✅' if has_nama else '❌'} "
            f"Face={'✅' if has_face_photo else '❌'} "
            f"Match={'✅' if nim_match else '❌'}"
        )

        result.total_time_ms = (time.perf_counter() - t_total_start) * 1000

        logger.info(
            f"Pipeline result: {result.status} | "
            f"NIM={result.nim_final} | Name={result.nama} | "
            f"[{indicators}] | {result.total_time_ms:.0f}ms"
        )

        return result
