#!/usr/bin/env python3
"""
Smart-Lab SV IPB — Webcam Test Script (v5: Low-Spec + DirectML Support)

Architecture:
 - Main thread: YOLO detection setiap frame (~40ms, no lag)
 - Stability check: OCR hanya trigger jika KTM stabil 3+ frame
 - Background thread: OCR/QR extraction (non-blocking)
 - Cooldown: 3s jeda setelah scan berhasil (cegah scan ganda)

Usage:
    python test_webcam.py
    python test_webcam.py --model models/best.pt --camera 1
    python test_webcam.py --low-spec          # Untuk laptop tanpa GPU / hardware lemah
    python test_webcam.py --low-spec --width 640 --height 480   # Resolusi lebih rendah

Controls:
    q      = Quit
    s      = Save current frame
    SPACE  = Pause/resume
    +/-    = Adjust confidence threshold
    o      = Force OCR now (bypass stability + cooldown)
    c      = Check-out (keluar lab)
"""

import argparse
import logging
import os
import sys
import time
import threading
from datetime import datetime

# ── macOS fix: pyzbar needs libzbar path ──
if sys.platform == "darwin":
    _homebrew_lib = "/opt/homebrew/lib"
    _existing = os.environ.get("DYLD_LIBRARY_PATH", "")
    if _homebrew_lib not in _existing:
        os.environ["DYLD_LIBRARY_PATH"] = f"{_homebrew_lib}:{_existing}"

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_webcam")

# ────────────────────────────────────────────────────────
# Color Palette (BGR)
# ────────────────────────────────────────────────────────
COLORS = {
    "qr_code":    (255, 165,   0),
    "text_nim":   (  0, 255, 127),
    "text_nama":  (255, 255,   0),
    "face_photo": (147, 20,  255),
}
COLOR_SUCCESS = (0, 255, 0)
COLOR_FAIL    = (0, 0, 255)
COLOR_WARN    = (0, 200, 255)
COLOR_INFO    = (255, 255, 255)
COLOR_BG      = (40, 40, 40)
COLOR_COOLDOWN = (255, 150, 50)
COLOR_DB_OK    = (0, 200, 100)    # green for DB verified
COLOR_DB_FAIL  = (80, 80, 255)    # red for DB not found

# ────────────────────────────────────────────────────────
# Scan Phase State Machine
# ────────────────────────────────────────────────────────
REQUIRED_LABELS = {"text_nim", "text_nama"}
MIN_STABLE_FRAMES = 3
ENROLLMENT_NAME_THRESHOLD = 0.70  # Enrollment butuh 70%+ nama cocok (anti salah NIM)
COOLDOWN_SECONDS = 3.0
FACE_VERIFY_SECONDS = 7.0   # durasi timer face verify
FACE_ENROLL_SECONDS = 10.0  # durasi timer enrollment (pertama kali)
IDENTITY_SHOW_SECONDS = 3.0 # delay sebelum face verify dimulai
COMPLETE_SHOW_SECONDS = 2.0 # durasi tampilkan success


class ScanPhase:
    """Fase scanning — state machine."""
    SCANNING = "scanning"           # 1. Menunggu KTM
    IDENTITY_FOUND = "identity"     # 2. Identitas ditemukan, tampilkan info
    FACE_ENROLL = "face_enroll"     # 3a. Pendaftaran wajah (pertama kali)
    FACE_VERIFY = "face_verify"     # 3b. Verifikasi wajah (sudah terdaftar)
    COMPLETE = "complete"           # 4. Scan berhasil
    COOLDOWN = "cooldown"           # 5. Jeda sebelum scan berikutnya


class StabilityTracker:
    """
    State machine untuk alur scanning KTM.
    Phase transitions:
      SCANNING → IDENTITY_FOUND → FACE_ENROLL/FACE_VERIFY → COMPLETE → COOLDOWN → SCANNING
    """
    def __init__(self, face_verify_enabled=True):
        self.consecutive_count = 0
        self.phase = ScanPhase.SCANNING
        self.phase_start = 0.0
        self.face_verify_enabled = face_verify_enabled
        self._needs_enrollment = False  # True jika belum ada encoding di DB
        self._face_failed = False       # True jika face verify gagal
        self._name_mismatch = False     # True jika nama OCR beda jauh dengan DB
        self._scan_rejection_reason = None  # Alasan scan ditolak (qr_missing, nim_mismatch, dll)

        # Scan results
        self.last_validated_nim = None
        self.last_validated_name = None
        self.db_result = None
        self.checkin_result = None
        self.face_result = None

    def update(self, detected_labels: set) -> bool:
        """Returns True if OCR should be triggered."""
        # Don't trigger OCR if not in SCANNING phase
        if self.phase != ScanPhase.SCANNING:
            self.consecutive_count = 0
            return False

        has_required = len(detected_labels & REQUIRED_LABELS) >= 1
        has_enough = len(detected_labels) >= 2

        if has_required and has_enough:
            self.consecutive_count += 1
        else:
            self.consecutive_count = 0

        return self.consecutive_count >= MIN_STABLE_FRAMES

    def enter_phase(self, phase: str):
        """Transition to a new phase."""
        logger.info(f"📌 Phase: {self.phase} → {phase}")
        self.phase = phase
        self.phase_start = time.time()
        self.consecutive_count = 0
        if phase == ScanPhase.SCANNING:
            self._face_failed = False
            self._scan_rejection_reason = None
            self._name_mismatch = False

    @property
    def phase_elapsed(self) -> float:
        return time.time() - self.phase_start

    @property
    def phase_remaining(self) -> float:
        """Remaining time for timed phases."""
        durations = {
            ScanPhase.IDENTITY_FOUND: IDENTITY_SHOW_SECONDS,
            ScanPhase.FACE_ENROLL: FACE_ENROLL_SECONDS,
            ScanPhase.FACE_VERIFY: FACE_VERIFY_SECONDS,
            ScanPhase.COMPLETE: COMPLETE_SHOW_SECONDS,
            ScanPhase.COOLDOWN: COOLDOWN_SECONDS,
        }
        d = durations.get(self.phase, 0)
        return max(0, d - self.phase_elapsed)

    def check_phase_timeout(self):
        """Auto-advance phases when their timer expires."""
        remaining = self.phase_remaining
        if remaining > 0:
            return
        if self.phase == ScanPhase.IDENTITY_FOUND:
            if not self.face_verify_enabled:
                self._do_checkin()
                self.enter_phase(ScanPhase.COMPLETE)
            elif self._needs_enrollment:
                self.enter_phase(ScanPhase.FACE_ENROLL)
            else:
                self.enter_phase(ScanPhase.FACE_VERIFY)
        elif self.phase == ScanPhase.FACE_ENROLL:
            # Enrollment timeout — check-in tanpa face
            logger.warning("⚠️ Enrollment timeout — check-in tanpa registrasi wajah")
            self._do_checkin()
            self.enter_phase(ScanPhase.COMPLETE)
        elif self.phase == ScanPhase.FACE_VERIFY:
            # Verify timeout — wajah TIDAK cocok → TOLAK
            logger.warning("❌ Face verify timeout — wajah tidak cocok, check-in DITOLAK")
            self._face_failed = True
            self.enter_phase(ScanPhase.COOLDOWN)
        elif self.phase == ScanPhase.COMPLETE:
            self.enter_phase(ScanPhase.COOLDOWN)
        elif self.phase == ScanPhase.COOLDOWN:
            self.enter_phase(ScanPhase.SCANNING)
            self.db_result = None
            self.checkin_result = None
            self.face_result = None

    def force_trigger(self):
        """Bypass — for 'o' key."""
        self.consecutive_count = MIN_STABLE_FRAMES
        self.phase = ScanPhase.SCANNING

    def reset_state(self):
        """Full reset — for 'r' key."""
        self.enter_phase(ScanPhase.SCANNING)
        self.db_result = None
        self.checkin_result = None
        self.face_result = None
        self.last_validated_nim = None
        self.last_validated_name = None
        self._face_failed = False
        self._needs_enrollment = False
        self._name_mismatch = False
        self._scan_rejection_reason = None

    def set_checkin_func(self, func):
        """Set the check_in function reference."""
        self._checkin_func = func

    def _do_checkin(self):
        """Execute check-in if NIM available."""
        nim = self.last_validated_nim
        if nim and hasattr(self, '_checkin_func'):
            ci_res = self._checkin_func(nim)
            self.checkin_result = ci_res
            logger.info(f"🏫 {ci_res['message']}")


def draw_center_text(display, msg, color, y_offset=0):
    """Draw large centered text with dark background."""
    h, w = display.shape[:2]
    (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    tx = (w - tw) // 2
    ty = h - 60 + y_offset
    cv2.rectangle(display, (tx - 15, ty - th - 12), (tx + tw + 15, ty + 12), (0, 0, 0), -1)
    cv2.putText(display, msg, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)


def draw_badge(display, text, color):
    """Draw status badge (top right)."""
    h, w = display.shape[:2]
    (stw, sth), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    sx = w - stw - 20
    cv2.rectangle(display, (sx - 10, 10), (w - 5, 10 + sth + 15), color, -1)
    cv2.putText(display, text, (sx, 10 + sth + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)


def draw_results(frame: np.ndarray, result, conf_threshold: float,
                 stability: StabilityTracker, fps: float) -> np.ndarray:
    """Draw bounding boxes, info panel, and phase-specific UX."""
    display = frame.copy()
    h, w = display.shape[:2]
    phase = stability.phase

    # ── Bounding boxes ──
    for label, bbox in result.bboxes.items():
        x1, y1, x2, y2 = bbox
        color = COLORS.get(label, (200, 200, 200))
        conf = result.confidences.get(label, 0.0)
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(display, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(display, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # ── Info Panel (kiri atas) ──
    panel_lines = []
    panel_lines.append(("SMART-LAB SV IPB", COLOR_INFO))
    panel_lines.append(("", None))

    # Detection info
    if result.detections_found:
        panel_lines.append((f"Detected: {', '.join(result.detections_found)}", COLOR_SUCCESS))
    else:
        panel_lines.append(("Arahkan KTM ke kamera...", COLOR_WARN))

    # Stability bar (only in SCANNING phase)
    if phase == ScanPhase.SCANNING:
        stable_pct = min(100, int(stability.consecutive_count / MIN_STABLE_FRAMES * 100))
        if stable_pct > 0:
            bar = "\u2588" * (stable_pct // 20) + "\u2591" * (5 - stable_pct // 20)
            panel_lines.append((f"Stability: {bar} {stable_pct}%", COLOR_WARN))

    panel_lines.append(("", None))

    # OCR results
    if result.nim_final:
        panel_lines.append((f"NIM: {result.nim_final}", COLOR_SUCCESS))
    if result.nama:
        panel_lines.append((f"NAMA: {result.nama}", COLOR_SUCCESS))

    # DB info
    db = stability.db_result
    if db and db.get("verified"):
        panel_lines.append(("", None))
        panel_lines.append((f"DB: {db['nama_db']}", COLOR_DB_OK))
        panel_lines.append((f"Prodi: {db.get('prodi', '-')}", COLOR_DB_OK))
        panel_lines.append((f"Angkatan: {db.get('angkatan', '-')}", COLOR_DB_OK))
        panel_lines.append((f"Match: {db['name_similarity']:.0%}", COLOR_DB_OK))
    elif db and not db.get("verified"):
        panel_lines.append(("", None))
        panel_lines.append((db.get("message", "DB: Not found"), COLOR_DB_FAIL))

    # Check-in status
    ci = stability.checkin_result
    if ci:
        color = COLOR_DB_OK if ci.get("success") else COLOR_WARN
        panel_lines.append((ci["message"], color))

    # Face result (verify or enroll)
    fr = stability.face_result
    if fr and phase in (ScanPhase.FACE_ENROLL, ScanPhase.FACE_VERIFY, ScanPhase.COMPLETE):
        if fr.get("verified"):
            panel_lines.append((f"Face: {fr.get('similarity', 0):.0%} COCOK", COLOR_DB_OK))
        elif fr.get("success") and fr.get("encoding") is not None:
            panel_lines.append(("Face: TERDAFTAR ✅", COLOR_DB_OK))
        elif fr.get("face_detected"):
            panel_lines.append((f"Face: {fr.get('similarity', 0):.0%}", COLOR_WARN))

    panel_lines.append(("", None))
    panel_lines.append((f"FPS: {fps:.1f}", COLOR_INFO))

    # Draw panel
    panel_h = len(panel_lines) * 22 + 16
    panel_w = 400
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), COLOR_BG, -1)
    cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)
    y = 20
    for text, color in panel_lines:
        if text and color:
            cv2.putText(display, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y += 22

    # ══════════════════════════════════════════════
    # Phase-specific badge + center text
    # ══════════════════════════════════════════════
    if phase == ScanPhase.SCANNING:
        count = stability.consecutive_count
        if count > 0:
            draw_badge(display, f"MEMINDAI... ({count}/{MIN_STABLE_FRAMES})", COLOR_WARN)
            draw_center_text(display,
                f"Memindai KTM... ({count}/{MIN_STABLE_FRAMES})", COLOR_WARN)
        else:
            draw_badge(display, "SCANNING...", COLOR_FAIL)

    elif phase == ScanPhase.IDENTITY_FOUND:
        remaining = stability.phase_remaining
        db = stability.db_result
        db_nama = db.get("nama_db", "-") if db else "-"
        draw_badge(display, "IDENTITAS DITEMUKAN", COLOR_DB_OK)
        if not stability.face_verify_enabled:
            draw_center_text(display,
                f"Identitas cocok! Memproses... ({remaining:.0f}s)",
                COLOR_DB_OK)
        elif stability._needs_enrollment:
            # Tampilkan nama DB besar agar user bisa konfirmasi visual
            draw_center_text(display,
                f"{db_nama}", (255, 200, 0), y_offset=-30)
            draw_center_text(display,
                f"Siap registrasi wajah... ({remaining:.0f}s)",
                (255, 200, 0), y_offset=10)
        else:
            draw_center_text(display,
                f"{db_nama}", COLOR_DB_OK, y_offset=-30)
            draw_center_text(display,
                f"Siapkan verifikasi... ({remaining:.0f}s)",
                COLOR_DB_OK, y_offset=10)

    elif phase == ScanPhase.FACE_ENROLL:
        remaining = stability.phase_remaining
        draw_badge(display, f"REGISTRASI WAJAH ({remaining:.0f}s)", (255, 200, 0))
        fr = stability.face_result
        if fr and fr.get("success"):
            draw_center_text(display, "WAJAH TERDAFTAR!", COLOR_DB_OK)
        elif fr and fr.get("face_detected"):
            draw_center_text(display,
                "Wajah terdeteksi — mendaftarkan...", (255, 200, 0))
        else:
            draw_center_text(display,
                f">> HADAP KE KAMERA << ({remaining:.0f}s)", (255, 200, 0))

    elif phase == ScanPhase.FACE_VERIFY:
        remaining = stability.phase_remaining
        draw_badge(display, f"VERIFIKASI WAJAH ({remaining:.0f}s)", COLOR_WARN)
        fr = stability.face_result
        if fr and fr.get("verified"):
            draw_center_text(display, "WAJAH COCOK!", COLOR_DB_OK)
        elif fr and fr.get("face_detected"):
            draw_center_text(display,
                f"Mencocokkan wajah... {fr['similarity']:.0%}", COLOR_WARN)
        else:
            draw_center_text(display,
                f">> HADAP KE KAMERA << ({remaining:.0f}s)", COLOR_WARN)

    elif phase == ScanPhase.COMPLETE:
        draw_badge(display, "CHECK-IN BERHASIL", COLOR_DB_OK)
        draw_center_text(display, "CHECK-IN BERHASIL!", COLOR_DB_OK)

    elif phase == ScanPhase.COOLDOWN:
        remaining = stability.phase_remaining
        rejection = stability._scan_rejection_reason
        if rejection == "qr_missing":
            draw_badge(display, "QR CODE TIDAK ADA", COLOR_DB_FAIL)
            draw_center_text(display,
                f"QR Code tidak terdeteksi! ({remaining:.0f}s)", COLOR_DB_FAIL)
        elif rejection == "nim_mismatch":
            draw_badge(display, "NIM TIDAK COCOK", COLOR_DB_FAIL)
            draw_center_text(display,
                f"NIM QR Dan OCR Belum Cocok ({remaining:.0f}s)", COLOR_DB_FAIL)
        elif rejection == "face_photo_missing":
            draw_badge(display, "FOTO KTM TIDAK ADA", COLOR_DB_FAIL)
            draw_center_text(display,
                f"Foto wajah di KTM tidak terdeteksi! ({remaining:.0f}s)", COLOR_DB_FAIL)
        elif stability._name_mismatch:
            draw_badge(display, "NAMA TIDAK COCOK", COLOR_DB_FAIL)
            draw_center_text(display,
                f"NAMA TIDAK COCOK — NIM SALAH? ({remaining:.0f}s)", COLOR_DB_FAIL)
        elif stability._face_failed:
            draw_badge(display, "VERIFIKASI GAGAL", COLOR_DB_FAIL)
            draw_center_text(display,
                f"VERIFIKASI WAJAH GAGAL! ({remaining:.0f}s)", COLOR_DB_FAIL)
        else:
            draw_badge(display, f"SELESAI ({remaining:.0f}s)", COLOR_COOLDOWN)

    return display


def main():
    parser = argparse.ArgumentParser(description="Smart-Lab KTM Scanner v5")
    parser.add_argument("--model", default="models/best.pt", help="Path to YOLO model")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--confidence", type=float, default=0.35, help="YOLO confidence")
    parser.add_argument("--width", type=int, default=1280, help="Capture width")
    parser.add_argument("--height", type=int, default=720, help="Capture height")
    parser.add_argument("--save-dir", default="captures", help="Save directory")
    parser.add_argument("--cooldown", type=float, default=3.0,
                        help="Cooldown after scan (seconds)")
    parser.add_argument("--no-face-verify", action="store_true",
                        help="Disable face verification")
    parser.add_argument(
        "--low-spec",
        action="store_true",
        help=(
            "Mode hardware rendah: YOLO interval 150ms, resolusi deteksi wajah 320px, "
            "QR 3 strategi saja, OCR tanpa angle-cls dan tanpa attempt 2. "
            "Gunakan ini jika FPS sangat rendah di laptop tanpa GPU."
        )
    )
    args = parser.parse_args()

    # ── Aktifkan Low-Spec Mode ──
    if args.low_spec:
        os.environ['SMARTLAB_LOW_SPEC'] = '1'
        logger.info("=" * 55)
        logger.info("   ⚡ LOW-SPEC MODE AKTIF")
        logger.info("   - YOLO: deteksi setiap 150ms (bukan setiap frame)")
        logger.info("   - InsightFace: det_size (320,320) [lebih ringan]")
        logger.info("   - QR Decode: 3 strategi saja (gray, binary, 2x)")
        logger.info("   - OCR: use_angle_cls=False, tanpa attempt ke-2")
        logger.info("=" * 55)
    else:
        os.environ.pop('SMARTLAB_LOW_SPEC', None)  # pastikan bersih
        logger.info("✅ Berjalan dalam mode NORMAL (full-spec).")

    # ── GPU Compatibility Check ──
    # Cek provider ONNX yang tersedia di sistem ini dan tampilkan status jelas.
    # Ini membantu pengguna tahu apakah DirectML/CUDA aktif atau tidak,
    # tanpa harus membaca log yang panjang.
    try:
        import onnxruntime as ort
        available_providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available_providers:
            logger.info("🟢 GPU Status: NVIDIA CUDA terdeteksi — akselerasi penuh aktif")
        elif 'DmlExecutionProvider' in available_providers:
            logger.info("🟡 GPU Status: AMD/Intel DirectML terdeteksi — akselerasi GPU aktif")
        else:
            logger.info("⚪ GPU Status: Tidak ada GPU akselerasi — berjalan via CPU")
            if sys.platform == "win32":
                logger.info("   ℹ️  Untuk AMD/Intel GPU: pip uninstall onnxruntime -y && pip install onnxruntime-directml")
                logger.info("   ℹ️  Jika DirectML gagal/tidak kompatibel: pip uninstall onnxruntime-directml -y && pip install onnxruntime")
    except ImportError:
        logger.warning("⚠️  onnxruntime tidak terinstall! Jalankan: pip install onnxruntime")

    global COOLDOWN_SECONDS
    COOLDOWN_SECONDS = args.cooldown

    # ── Pre-warm PaddleOCR ──
    logger.info("Pre-loading PaddleOCR...")
    try:
        from ml.extractor import _get_paddle_ocr
        _get_paddle_ocr()
        logger.info("✅ PaddleOCR ready")
    except Exception as e:
        logger.warning(f"PaddleOCR pre-load failed: {e}")

    # ── Verify pyzbar ──
    try:
        from pyzbar.pyzbar import decode as _test
        logger.info("✅ pyzbar ready")
    except Exception as e:
        logger.warning(f"⚠️ pyzbar unavailable: {e}")

    # ── Init Database ──
    from db.database import (
        init_db, verify_student, check_in, check_out,
        reset_all_peminjaman, save_face_encoding, load_face_encoding,
    )
    init_db()

    # ── Init Face Verifier ──
    face_verifier = None
    if not args.no_face_verify:
        from ml.face_verify import FaceVerifier
        face_verifier = FaceVerifier()
        logger.info("✅ Face verification enabled")
    else:
        logger.info("⚠️ Face verification disabled")

    # ── Init Pipeline ──
    from ml.pipeline import KTMPipeline, ScanResult

    pipeline = KTMPipeline(model_path=args.model, confidence_threshold=args.confidence)
    if not pipeline.is_ready():
        logger.error("❌ Model not found. Running DEMO MODE.")

    # ── Init Webcam ──
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error(f"❌ Cannot open camera {args.camera}!")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    logger.info(f"✅ Camera: {int(cap.get(3))}x{int(cap.get(4))}")

    os.makedirs(args.save_dir, exist_ok=True)

    # ──────────────────────────────────────────────────
    # Background OCR Thread + Stability Tracker
    # ──────────────────────────────────────────────────
    stability = StabilityTracker(face_verify_enabled=(face_verifier is not None))
    stability.set_checkin_func(check_in)
    ocr_lock = threading.Lock()
    ocr_cached = ScanResult()
    ocr_busy = False
    ocr_snapshot = None
    ocr_trigger = threading.Event()
    should_quit = threading.Event()

    def ocr_worker():
        nonlocal ocr_cached, ocr_busy, ocr_snapshot
        while not should_quit.is_set():
            ocr_trigger.wait(timeout=0.5)
            if should_quit.is_set():
                break
            if not ocr_trigger.is_set():
                continue
            ocr_trigger.clear()

            with ocr_lock:
                snap = ocr_snapshot.copy() if ocr_snapshot is not None else None
                ocr_busy = True

            if snap is None:
                with ocr_lock:
                    ocr_busy = False
                continue

            try:
                result = pipeline.process_frame(snap)
                with ocr_lock:
                    ocr_cached = result
                    ocr_busy = False

                logger.info(
                    f"📋 OCR: NIM={result.nim_final} | "
                    f"Name={result.nama} | Match={result.nim_match} | "
                    f"{result.total_time_ms:.0f}ms"
                )

                # ── DB Lookup (hanya jika 4 indikator lengkap + match) ──
                if result.success:
                    db_res = verify_student(result.nim_final, result.nama)
                    stability.db_result = db_res
                    stability.last_validated_nim = result.nim_final
                    stability.last_validated_name = result.nama

                    if db_res["verified"]:
                        nim = result.nim_final
                        name_sim = db_res.get("name_similarity", 0)
                        if face_verifier:
                            # Check if encoding exists in DB
                            db_encoding = load_face_encoding(nim)
                            if db_encoding is not None:
                                # Sudah punya encoding → langsung verify
                                face_verifier.set_reference_from_encoding(db_encoding)
                                stability._needs_enrollment = False
                                logger.info(f"👤 Face encoding loaded dari DB untuk {nim}")
                            else:
                                # Belum ada encoding → cek nama dulu sebelum enrollment
                                if name_sim >= ENROLLMENT_NAME_THRESHOLD:
                                    stability._needs_enrollment = True
                                    logger.info(
                                        f"👤 Enrollment OK: nama cocok {name_sim:.0%} "
                                        f"(OCR='{result.nama}' vs DB='{db_res.get('nama_db')}')")
                                else:
                                    # Nama terlalu beda → tolak enrollment!
                                    stability._needs_enrollment = False
                                    stability._name_mismatch = True
                                    logger.warning(
                                        f"🚫 Enrollment DITOLAK: nama hanya {name_sim:.0%} "
                                        f"(OCR='{result.nama}' vs DB='{db_res.get('nama_db')}') "
                                        f"— kemungkinan NIM salah baca!")
                                    stability.enter_phase(ScanPhase.COOLDOWN)
                                    continue

                        # Enter IDENTITY_FOUND phase
                        stability.enter_phase(ScanPhase.IDENTITY_FOUND)
                    else:
                        stability.checkin_result = None
                        logger.warning(f"🚫 {db_res['message']}")
                        # NIM not found — go to cooldown
                        stability.enter_phase(ScanPhase.COOLDOWN)

                elif result.status == "qr_missing":
                    # KTM tanpa QR code — DITOLAK
                    logger.warning(
                        f"🚫 QR Code tidak terdeteksi! NIM OCR={result.nim_ocr}, "
                        f"Nama={result.nama} — scan DITOLAK (butuh QR)")
                    stability._scan_rejection_reason = "qr_missing"
                    stability.enter_phase(ScanPhase.COOLDOWN)
                elif result.status == "nim_mismatch":
                    # QR dan OCR beda — DITOLAK
                    logger.warning(
                        f"🚫 NIM tidak cocok! QR={result.nim_qr} vs "
                        f"OCR={result.nim_ocr} — scan DITOLAK")
                    stability._scan_rejection_reason = "nim_mismatch"
                    stability.enter_phase(ScanPhase.COOLDOWN)
                elif result.status == "face_photo_missing":
                    logger.warning(
                        f"🚫 Foto wajah pada KTM tidak terdeteksi — "
                        f"pastikan KTM terlihat jelas!")
                    stability._scan_rejection_reason = "face_photo_missing"
                    stability.enter_phase(ScanPhase.COOLDOWN)

            except Exception as e:
                logger.error(f"OCR error: {e}")
                with ocr_lock:
                    ocr_busy = False

    ocr_thread = threading.Thread(target=ocr_worker, daemon=True)
    ocr_thread.start()

    # ── Main Loop ──
    conf_threshold = args.confidence
    paused = False
    frame_count = 0
    fps_time = time.time()
    fps = 0.0

    # [low-spec] Interval YOLO: hanya jalankan deteksi setiap N detik
    # Mode normal  = setiap frame (~33ms @ 30fps)
    # Mode low-spec = setiap 150ms (maks 7x/detik, hemat CPU)
    YOLO_INTERVAL = 0.15 if args.low_spec else 0.0
    last_yolo_time = 0.0
    last_detections: dict = {}  # Cache hasil deteksi terakhir

    logger.info("Starting scan. Press 'q' to quit.")
    print(f"\n  [q] Quit | [s] Save | [SPACE] Pause | [+/-] Confidence")
    print(f"  [o] Force OCR | [c] Check-out | [r] Reset peminjaman")
    print(f"  [f] Reset face encoding | Stability: {MIN_STABLE_FRAMES} frames")
    if args.low_spec:
        print(f"  ⚡ LOW-SPEC MODE: YOLO tiap {YOLO_INTERVAL*1000:.0f}ms, OCR dioptimasi\n")
    else:
        print()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        if not paused and pipeline.is_ready():
            pipeline.confidence_threshold = conf_threshold

            # Auto-advance timed phases
            stability.check_phase_timeout()

            # ══════════════════════════════════════════════
            # PATH A: YOLO + OCR (only during SCANNING)
            # ══════════════════════════════════════════════
            if stability.phase == ScanPhase.SCANNING:
                # [low-spec] Interval YOLO: skip deteksi jika belum waktunya
                now = time.time()
                if now - last_yolo_time >= YOLO_INTERVAL:
                    detections = pipeline.detect(frame)
                    last_detections = pipeline.crop_detections(frame, detections) if detections else {}
                    last_yolo_time = now
                crops = last_detections

                display_result = ScanResult()
                display_result.detections_found = list(crops.keys())
                display_result.confidences = {k: round(v.confidence, 3) for k, v in crops.items()}
                display_result.bboxes = {k: v.bbox for k, v in crops.items()}

                with ocr_lock:
                    cached = ocr_cached
                    busy = ocr_busy

                display_result.nim_qr = cached.nim_qr
                display_result.nim_ocr = cached.nim_ocr
                display_result.nim_final = cached.nim_final
                display_result.nama = cached.nama
                display_result.nim_match = cached.nim_match
                display_result.validation_detail = cached.validation_detail
                display_result.success = cached.success
                if cached.success:
                    display_result.status = cached.status
                display_result.inference_time_ms = cached.inference_time_ms
                display_result.total_time_ms = cached.total_time_ms

                detected_set = set(crops.keys())
                should_ocr = stability.update(detected_set)
                if should_ocr and not busy:
                    logger.info(f"🔍 KTM stabil → triggering OCR...")
                    stability.consecutive_count = 0
                    with ocr_lock:
                        ocr_snapshot = frame.copy()
                    ocr_trigger.set()

                if busy:
                    cv2.putText(display, "OCR Processing...", (10, display.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WARN, 2)

            # ══════════════════════════════════════════════
            # PATH B: FACE ENROLLMENT (pertama kali, YOLO OFF)
            # ══════════════════════════════════════════════
            elif stability.phase == ScanPhase.FACE_ENROLL:
                display_result = ScanResult()
                with ocr_lock:
                    cached = ocr_cached
                display_result.nim_final = cached.nim_final
                display_result.nama = cached.nama
                display_result.success = cached.success

                if face_verifier:
                    enroll_res = face_verifier.enroll(frame)
                    stability.face_result = enroll_res

                    # Draw face bbox (kuning untuk enrollment)
                    if enroll_res.get("face_detected") and enroll_res.get("face_bbox"):
                        fx, fy, fw, fh = enroll_res["face_bbox"]
                        cv2.rectangle(display, (fx, fy), (fx+fw, fy+fh), (255, 200, 0), 3)
                        cv2.putText(display, "Mendaftarkan...", (fx, fy - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2, cv2.LINE_AA)

                    # Wajah berhasil di-capture → simpan encoding ke DB
                    if enroll_res.get("success") and enroll_res.get("encoding") is not None:
                        nim = stability.last_validated_nim
                        if nim:
                            save_face_encoding(nim, enroll_res["encoding"])
                            logger.info(f"✅ Face enrolled dan disimpan ke DB untuk {nim}")
                        stability._do_checkin()
                        stability.enter_phase(ScanPhase.COMPLETE)

            # ══════════════════════════════════════════════
            # PATH C: FACE VERIFY (sudah terdaftar, YOLO OFF)
            # ══════════════════════════════════════════════
            elif stability.phase == ScanPhase.FACE_VERIFY:
                display_result = ScanResult()
                with ocr_lock:
                    cached = ocr_cached
                display_result.nim_final = cached.nim_final
                display_result.nama = cached.nama
                display_result.success = cached.success

                if face_verifier and face_verifier.has_reference:
                    face_res = face_verifier.verify(frame)
                    stability.face_result = face_res

                    # Draw face bbox
                    if face_res["face_detected"] and face_res["face_bbox"]:
                        fx, fy, fw, fh = face_res["face_bbox"]
                        fc = COLOR_DB_OK if face_res["verified"] else COLOR_WARN
                        cv2.rectangle(display, (fx, fy), (fx+fw, fy+fh), fc, 3)
                        label = f"Face: {face_res['similarity']:.0%}"
                        cv2.putText(display, label, (fx, fy - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, fc, 2, cv2.LINE_AA)

                    # Face matched! → check-in + COMPLETE
                    if face_res.get("verified"):
                        logger.info(f"✅ Wajah cocok: {face_res['similarity']:.0%}")
                        stability._do_checkin()
                        stability.enter_phase(ScanPhase.COMPLETE)

            # ══════════════════════════════════════════════
            # PATH C: NO PROCESSING (IDENTITY_FOUND / COMPLETE / COOLDOWN)
            # ══════════════════════════════════════════════
            else:
                display_result = ScanResult()
                with ocr_lock:
                    cached = ocr_cached
                display_result.nim_final = cached.nim_final
                display_result.nama = cached.nama
                display_result.success = cached.success

            display = draw_results(display, display_result, conf_threshold,
                                   stability, fps)

        elif paused:
            cv2.putText(display, "PAUSED", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_WARN, 3, cv2.LINE_AA)

        # FPS
        frame_count += 1
        elapsed = time.time() - fps_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_time = time.time()

        cv2.imshow("Smart-Lab SV IPB — KTM Scanner", display)

        # ── Key handling ──
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fn = os.path.join(args.save_dir, f"ktm_capture_{ts}.jpg")
            cv2.imwrite(fn, frame)
            logger.info(f"📸 Saved: {fn}")
        elif key == ord(" "):
            paused = not paused
        elif key == ord("+") or key == ord("="):
            conf_threshold = min(0.95, conf_threshold + 0.05)
            logger.info(f"Threshold: {conf_threshold:.0%}")
        elif key == ord("-"):
            conf_threshold = max(0.05, conf_threshold - 0.05)
            logger.info(f"Threshold: {conf_threshold:.0%}")
        elif key == ord("o"):
            stability.force_trigger()
            logger.info("🔄 Force OCR!")
        elif key == ord("c"):
            nim = stability.last_validated_nim
            if nim:
                co_res = check_out(nim)
                logger.info(f"📤 {co_res['message']}")
                stability.checkin_result = co_res
            else:
                logger.warning("⚠️ Belum ada NIM yang di-scan")
        elif key == ord("r"):
            res = reset_all_peminjaman()
            logger.info(f"🔄 {res['message']}")
            stability.reset_state()
            with ocr_lock:
                ocr_cached = ScanResult()
            logger.info("🔄 State di-reset — siap scan ulang")
        elif key == ord("f"):
            from db.database import get_connection
            with get_connection() as conn:
                conn.execute('UPDATE mahasiswa SET face_encoding = NULL')
            res2 = reset_all_peminjaman()
            logger.info(f"🧹 Face encoding di-reset untuk semua mahasiswa")
            logger.info(f"🔄 {res2['message']}")
            stability.reset_state()
            with ocr_lock:
                ocr_cached = ScanResult()
            if face_verifier:
                face_verifier.clear_reference()

    # ── Cleanup ──
    should_quit.set()
    ocr_trigger.set()
    ocr_thread.join(timeout=3)
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Camera released. Goodbye!")


if __name__ == "__main__":
    main()
