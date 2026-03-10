"""
Smart-Lab SV IPB — Face Liveness Detection (Anti-Spoofing)
Modul ini bertugas mendeteksi apakah wajah yang menghadap kamera
adalah wajah manusia asli (3D) atau hasil manipulasi (Foto Cetak / Layar HP).

Metode: Hardware-free 2D Static Texture Heuristics.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FaceLivenessDetector:
    def __init__(self, blur_threshold: float = 65.0, glare_threshold: float = 0.05):
        """
        :param blur_threshold: Nilai minimal variansi Laplacian untuk tidak dianggap blur.
                               Foto cetakan/layar HP murah cenderung kurang tajam di mikrotekstur.
        :param glare_threshold: Persentase wajar area putih terang (Pijaran Layar).
                                Layar HP menghasilkan specular reflection / pantulan silau berlebih.
        """
        self.blur_threshold = blur_threshold
        self.glare_threshold = glare_threshold

    def analyze_liveness(self, face_img: np.ndarray) -> dict:
        """
        Menganalisa crop gambar wajah menggunakan heuristik statik tekstur.
        """
        if face_img is None or face_img.size == 0:
             return {"is_real": False, "message": "Gambar wajah kosong atau tidak valid."}

        # 1. Laplasian Variance (Blur Detection)
        # Menghitung variansi intensitas laplacian (ketajaman garis/edge pada wajah)
        # Wajah palsu (foto 2D) atau layar HP seringkali mengalami loss mikrotekstur
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blurry = laplacian_var < self.blur_threshold

        # 2. Specular Reflection (Glare / Overexposure Detection)
        # Layar HP memancarkan cahaya lampu latarnya sendiri (Backlight),
        # sehingga saat dihadapkan pada Webcam, akan memicu hotspot overexposure.
        hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2] # Value/Brightness channel
        
        # Hitung densitas piksel yang super cerah (>240)
        overexposed_pixels = np.sum(v_channel > 240)
        total_pixels = face_img.shape[0] * face_img.shape[1]
        glare_ratio = overexposed_pixels / float(total_pixels)
        is_glaring = glare_ratio > self.glare_threshold

        # --- Keputusan Akhir Heuristik ---
        is_real = not (is_blurry or is_glaring)

        # Buat deskripsi pesan log untuk debug
        messages = []
        if is_blurry: messages.append(f"Terindikasi Blur/Foto Cetak (Var: {laplacian_var:.1f})")
        if is_glaring: messages.append(f"Terindikasi Layar HP/Glare (R: {glare_ratio:.3f})")
        
        status_msg = "Aman (Wajah Asli)" if is_real else " | ".join(messages)
        
        logger.debug(
            f"[Liveness] Real: {is_real} | Var: {laplacian_var:.2f} "
            f"| Glare: {glare_ratio:.4f} | Msg: {status_msg}"
        )

        return {
            "is_real": bool(is_real),
            "score_blur": float(laplacian_var),
            "score_glare": float(glare_ratio),
            "is_blurry": bool(is_blurry),
            "is_glaring": bool(is_glaring),
            "message": status_msg
        }

# Buat instansi singleton untuk diimpor langsung
detector = FaceLivenessDetector()
