"""
Smart-Lab SV IPB — Face Liveness Detection (Anti-Spoofing)
Modul ini mendeteksi apakah wajah yang menghadap kamera adalah wajah asli (3D)
atau hasil manipulasi (Foto Cetak / Layar HP).

══════════════════════════════════════════════════════════════
 METODE: 3-Signal Majority Voting (lebih robust dari 2-sinyal)
══════════════════════════════════════════════════════════════
 Signal 1: GLARE      – rasio piksel overexposed (layar HP = backlit tinggi)
 Signal 2: SHARPNESS  – Laplacian variance (palsu = mikrotekstur rendah)
 Signal 3: TEXTURE    – Entropi LBP (kulit asli = variasi tinggi)

 Keputusan FAKE: jika 2 dari 3 signal mendeteksi anomali (majority vote)
 Keputusan REAL: jika 0 atau 1 signal yang gagal

 Contoh kasus:
 ┌─────────────────────────────┬───────┬───────┬───────┬──────────┐
 │ Skenario                    │ Glare │ Blur  │ Flat  │ Keputusan│
 ├─────────────────────────────┼───────┼───────┼───────┼──────────┤
 │ Wajah asli, cahaya normal   │  ❌   │  ❌   │  ❌   │ ✅ REAL  │
 │ Wajah asli, lampu terang    │  ❌   │  ❌   │  ❌   │ ✅ REAL  │
 │ Wajah asli, webcam blur     │  ❌   │  ✅   │  ❌   │ ✅ REAL  │
 │ Layar HP (screen spoof)     │  ✅   │  ✅*  │  ✅   │ ❌ FAKE  │
 │ Foto cetak matte            │  ❌   │  ✅   │  ✅   │ ❌ FAKE  │
 │ Foto glossy (kertas foto)   │  ✅*  │  ❌   │  ✅   │ ❌ FAKE  │
 └─────────────────────────────┴───────┴───────┴───────┴──────────┘
 * = tidak selalu, tapi biasanya
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FaceLivenessDetector:
    def __init__(
        self,
        # Threshold masing-masing signal
        blur_threshold: float = 35.0,
        # Laplacian var < 35 → suspiciously smooth
        # Webcam laptop normal: 40–150 | Foto cetak/layar: <30

        glare_threshold: float = 0.12,
        # Overexposed pixels > 12% → screen backlight detected
        # Lampu ruangan normal: <8% | Layar HP aktif: >15%

        texture_threshold: float = 2.5,
        # LBP entropy < 2.5 → texture terlalu seragam (bukan kulit asli)
        # Wajah asli: 3.0–4.0 | Foto/layar: 1.0–2.5

        min_signals_to_reject: int = 2,
        # Minimal 2 dari 3 signal harus gagal → FAKE
    ):
        self.blur_threshold = blur_threshold
        self.glare_threshold = glare_threshold
        self.texture_threshold = texture_threshold
        self.min_signals_to_reject = min_signals_to_reject

    def _lbp_entropy(self, gray: np.ndarray) -> float:
        """
        Hitung entropi LBP (Local Binary Pattern) sebagai ukuran kekayaan mikrotekstur.

        LBP membandingkan tiap piksel dengan tetangganya → menghasilkan kode biner.
        Distribusi kode yang beragam (entropy tinggi) = tekstur bervariasi = kulit asli.
        Distribusi kode terpusat (entropy rendah) = tekstur seragam = foto/layar.

        Implementasi fast 4-neighbor numpy (tanpa loop Python):
          code = n_left + 2*n_right + 4*n_up + 8*n_down  → 16 kemungkinan nilai
        """
        h, w = gray.shape
        if h < 3 or w < 3:
            return 0.0

        center = gray[1:h-1, 1:w-1].astype(np.float32)

        # Toleransi ±3 untuk mengurangi sensitivitas terhadap noise kamera
        threshold = 3.0
        n_left  = (gray[1:h-1, 0:w-2].astype(np.float32) >= center + threshold).view(np.uint8)
        n_right = (gray[1:h-1, 2:w  ].astype(np.float32) >= center + threshold).view(np.uint8)
        n_up    = (gray[0:h-2, 1:w-1].astype(np.float32) >= center + threshold).view(np.uint8)
        n_down  = (gray[2:h,   1:w-1].astype(np.float32) >= center + threshold).view(np.uint8)

        code = n_left + 2 * n_right + 4 * n_up + 8 * n_down

        # Histogram distribusi pola (16 bin)
        hist, _ = np.histogram(code, bins=16, range=(0, 16))
        hist_norm = hist / float(hist.sum() + 1e-10)

        # Shannon entropy: tinggi = distribusi merata = banyak pola berbeda = kulit asli
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
        return float(entropy)

    def analyze_liveness(self, face_img: np.ndarray) -> dict:
        """
        Analisis liveness wajah menggunakan 3-signal majority voting.

        Returns:
            dict dengan kunci:
              is_real        : bool   – True jika wajah dianggap asli
              score_blur     : float  – nilai Laplacian variance (semakin tinggi semakin tajam)
              score_glare    : float  – rasio piksel overexposed (0.0–1.0)
              score_texture  : float  – LBP entropy (semakin tinggi semakin real)
              fail_count     : int    – jumlah signal yang gagal (0–3)
              message        : str    – deskripsi hasil
        """
        if face_img is None or face_img.size == 0:
            return {
                "is_real": False, "score_blur": 0, "score_glare": 0,
                "score_texture": 0, "fail_count": 3,
                "message": "Gambar wajah kosong atau tidak valid.",
            }

        # Pastikan crop cukup besar untuk analisis yang valid
        h, w = face_img.shape[:2]
        if h < 32 or w < 32:
            logger.debug("[Liveness] Face crop terlalu kecil, dilewati.")
            return {
                "is_real": True, "score_blur": 0, "score_glare": 0,
                "score_texture": 0, "fail_count": 0,
                "message": "Crop wajah terlalu kecil untuk dianalisis.",
            }

        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)

        # ── Signal 1: GLARE — Overexposure / Backlit Screen ──────────────────
        # Layar HP memancarkan cahaya sendiri → banyak piksel overexposed (V > 240)
        # Wajah asli dengan lampu normal: <8% | Layar HP aktif: >15–40%
        v_channel   = hsv[:, :, 2]
        glare_ratio = float(np.sum(v_channel > 240)) / float(h * w)
        signal_glare = glare_ratio > self.glare_threshold

        # ── Signal 2: SHARPNESS — Laplacian Variance ─────────────────────────
        # Foto cetak/layar HP kehilangan mikrotekstur → variansi Laplacian rendah
        # Webcam laptop normal: 40–150 | Foto matte/layar HP: <30
        lap_var     = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        signal_blur = lap_var < self.blur_threshold

        # ── Signal 3: MICRO-TEXTURE — LBP Entropy ────────────────────────────
        # Kulit manusia nyata punya mikrotekstur kaya (pori, garis halus, dll)
        # Foto cetak dan layar HP menghasilkan pola piksel yang lebih seragam
        lbp_entropy = self._lbp_entropy(gray)
        signal_flat = lbp_entropy < self.texture_threshold

        # ── Majority Voting Decision ──────────────────────────────────────────
        failures = int(signal_glare) + int(signal_blur) + int(signal_flat)
        is_fake  = failures >= self.min_signals_to_reject
        is_real  = not is_fake

        # Build informative status message
        fail_details = []
        if signal_glare: fail_details.append(f"Glare({glare_ratio:.3f}>{self.glare_threshold})")
        if signal_blur:  fail_details.append(f"Blur({lap_var:.1f}<{self.blur_threshold})")
        if signal_flat:  fail_details.append(f"FlatTex({lbp_entropy:.2f}<{self.texture_threshold})")

        if is_real:
            status_msg = (
                f"Aman [{failures}/3 gagal] — "
                f"Var:{lap_var:.1f} | Glare:{glare_ratio:.3f} | Tex:{lbp_entropy:.2f}"
            )
        else:
            status_msg = (
                f"Terindikasi Palsu [{failures}/3 gagal]: " + " & ".join(fail_details)
            )

        logger.debug(
            f"[Liveness] Real={is_real} | fails={failures}/3 | "
            f"Blur={signal_blur}({lap_var:.1f}) | "
            f"Glare={signal_glare}({glare_ratio:.4f}) | "
            f"FlatTex={signal_flat}({lbp_entropy:.2f})"
        )

        return {
            "is_real":       bool(is_real),
            "score_blur":    float(lap_var),
            "score_glare":   float(glare_ratio),
            "score_texture": float(lbp_entropy),
            "is_blurry":     bool(signal_blur),
            "is_glaring":    bool(signal_glare),
            "is_flat_texture": bool(signal_flat),
            "fail_count":    failures,
            "message":       status_msg,
        }


# Singleton — diimpor langsung oleh face_verify.py
detector = FaceLivenessDetector()
