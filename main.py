"""
Smart-Lab SV IPB — FastAPI Backend
REST API + WebSocket endpoint untuk KTM scanning.

NFR Refactored:
- [S1] CORS origins dari environment variable
- [S2] Input validation (NIM format, file size)
- [S3] File size limit (MAX_UPLOAD_SIZE_MB)
- [S4] No internal error details leaked
- [S5] Admin API Key authentication (X-Admin-Key header)
- [R3] WebSocket DB integration
- [O1] Request timing in response headers

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Admin Key:
    Set environment variable: export ADMIN_API_KEY=<your-secret-key>
"""

import asyncio
import base64
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

from pydantic import BaseModel

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader

from ml.pipeline import KTMPipeline, ScanResult
from ml.face_verify import FaceVerifier
from db.database import (
    init_db, verify_student, check_in, check_out,
    get_active_peminjaman, reset_all_peminjaman, NIM_PATTERN,
    save_face_encoding, load_face_encoding, delete_face_encoding
)

# ────────────────────────────────────────────────────────
# Constants [M1]
# ────────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB = 10  # [S3] Batas ukuran file upload
MAX_UPLOAD_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
WS_MIN_INTERVAL_MS = 300  # Debounce interval WebSocket

# CORS origins — dari env var, default restrictive [S1]
ALLOWED_ORIGINS = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000"
).split(",")

# Admin API Key [S5] — from env var, WAJIB diset di production!
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "CHANGE-ME-IN-PRODUCTION")
_api_key_header = APIKeyHeader(name="X-Admin-Key", auto_error=False)


async def require_admin_key(api_key: Optional[str] = Security(_api_key_header)):
    """
    [S5] Dependency Guard untuk endpoint admin.
    Membutuhkan header: X-Admin-Key: <ADMIN_API_KEY>
    """
    if api_key != ADMIN_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized. Header X-Admin-Key diperlukan."
        )


# ────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("smartlab-api")

# ────────────────────────────────────────────────────────
# Global pipeline (singleton)
# ────────────────────────────────────────────────────────
pipeline: Optional[KTMPipeline] = None
face_verifier: Optional[FaceVerifier] = None
face_verifier_lock = asyncio.Lock()


# ────────────────────────────────────────────────────────
# Pydantic Schemas (fastapi-pro: API contract first)
# ────────────────────────────────────────────────────────
class FaceRequest(BaseModel):
    """Request body untuk face enroll/verify endpoint."""
    nim: str
    nama: Optional[str] = None  # Tambahkan nama dari hasil OCR untuk name match guard
    image_base64: str  # Base64 encoded JPEG/PNG dari kamera Frontend

class FaceResponse(BaseModel):
    status: str
    nim: Optional[str] = None
    nama: Optional[str] = None
    similarity: Optional[float] = None
    message: Optional[str] = None
    checkin: Optional[Dict[str, Any]] = None
    processing_time_ms: float



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize ML pipeline on startup, cleanup on shutdown."""
    global pipeline
    logger.info("🚀 Starting Smart-Lab ML Engine...")
    init_db()
    logger.info("✅ Database ready")
    pipeline = KTMPipeline(model_path="models/best.pt")
    if pipeline.is_ready():
        logger.info("✅ Pipeline ready!")
    else:
        logger.warning("⚠️ Pipeline loaded but model NOT FOUND. Will return empty results.")
    yield
    logger.info("🛑 Shutting down Smart-Lab ML Engine.")
    pipeline = None


# ────────────────────────────────────────────────────────
# FastAPI App
# ────────────────────────────────────────────────────────
app = FastAPI(
    title="Smart-Lab SV IPB — ML Engine",
    description="Real-time KTM scanning via YOLO + OCR pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — restrictive by default [S1]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


# ────────────────────────────────────────────────────────
# Input Validation Helpers [S2]
# ────────────────────────────────────────────────────────
_NIM_RE = re.compile(NIM_PATTERN)


def _validate_nim(nim: str) -> str:
    """Validate dan sanitize NIM. Raises HTTPException jika invalid."""
    nim = nim.strip().upper()
    if not _NIM_RE.match(nim):
        raise HTTPException(
            status_code=400,
            detail=f"Format NIM tidak valid: '{nim}'. Expected: J + 6-12 alphanumeric chars."
        )
    return nim


def _ensure_pipeline_ready():
    """Guard: pastikan pipeline siap. Raises HTTPException jika belum."""
    if not pipeline or not pipeline.is_ready():
        raise HTTPException(
            status_code=503,
            detail="ML pipeline not ready. Model might be missing."
        )


def _get_face_verifier() -> FaceVerifier:
    """Lazy-load FaceVerifier singleton. Model InsightFace di-load sekali."""
    global face_verifier
    if face_verifier is None:
        logger.info("🛠 Inisialisasi FaceVerifier untuk API...")
        face_verifier = FaceVerifier()
    return face_verifier


# ────────────────────────────────────────────────────────
# Helper
# ────────────────────────────────────────────────────────
def decode_base64_image(data: str) -> Optional[np.ndarray]:
    """Decode base64-encoded image ke numpy array (BGR)."""
    try:
        if "," in data:
            data = data.split(",", 1)[1]
        img_bytes = base64.b64decode(data)
        # [S3] Size check for base64 too
        if len(img_bytes) > MAX_UPLOAD_BYTES:
            logger.warning(f"Base64 image too large: {len(img_bytes)} bytes")
            return None
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.error(f"Failed to decode image: {type(e).__name__}")  # [S4] No detail leak
        return None


def _process_scan_result(result: ScanResult) -> dict:
    """
    Process scan result: verify DB identity.
    [M4] Single source of truth untuk scan→verify logic.
    [SECURITY] TIDAK auto check-in. Frontend harus panggil /api/face/* dulu.
    """
    response = result.to_dict()

    if result.nim_final and result.nama:
        db_res = verify_student(result.nim_final, result.nama)
        response["db_verified"] = db_res["verified"]
        response["db_nama"] = db_res.get("nama_db")
        response["db_prodi"] = db_res.get("prodi")
        response["db_message"] = db_res.get("message")

        if db_res["verified"]:
            # Cek apakah sudah punya face encoding di DB
            db_encoding = load_face_encoding(result.nim_final)
            response["action_required"] = (
                "face_verify" if db_encoding is not None else "face_enroll"
            )

    return response


# ────────────────────────────────────────────────────────
# REST Endpoints
# ────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": pipeline.is_ready() if pipeline else False,
        "service": "smart-lab-ml-engine",
    }


@app.post("/api/scan")
async def scan_image(file: UploadFile = File(...)):
    """
    REST endpoint untuk scan single image.
    [S3] File size limit enforced.
    """
    _ensure_pipeline_ready()

    # [S3] Read with size limit
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File terlalu besar. Maksimal {MAX_UPLOAD_SIZE_MB}MB."
        )

    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Process + DB verify [M4] — via shared helper
    t_start = time.perf_counter()
    result = pipeline.process_frame(frame)
    response = _process_scan_result(result)
    response["processing_time_ms"] = round((time.perf_counter() - t_start) * 1000, 1)

    return response


# ────────────────────────────────────────────────────────
# WebSocket Endpoint [R3] — now with DB integration
# ────────────────────────────────────────────────────────
@app.websocket("/ws/scan")
async def websocket_scan(ws: WebSocket):
    """
    WebSocket endpoint untuk real-time KTM scanning.
    [R3] Sekarang include DB verify + check-in.
    """
    await ws.accept()
    client_id = id(ws)
    logger.info(f"🔌 WebSocket connected: {client_id}")

    last_process_time = 0.0

    try:
        while True:
            raw_data = await ws.receive_text()

            # Debounce
            now = time.time() * 1000
            if now - last_process_time < WS_MIN_INTERVAL_MS:
                await ws.send_json({"status": "skipped", "reason": "debounce"})
                continue

            # Decode image
            try:
                msg = json.loads(raw_data)
                image_data = msg.get("image", msg.get("frame", raw_data))
            except (json.JSONDecodeError, TypeError):
                image_data = raw_data

            frame = decode_base64_image(image_data)

            if frame is None:
                await ws.send_json({"status": "error", "error": "Failed to decode image."})
                continue

            # Process + DB verify [R3][M4]
            if pipeline and pipeline.is_ready():
                result = await asyncio.to_thread(pipeline.process_frame, frame)
                response = _process_scan_result(result)  # shared logic
            else:
                response = {"status": "model_not_loaded", "error": "ML model not available."}

            await ws.send_json(response)
            last_process_time = time.time() * 1000

    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket disconnected: {client_id}")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {type(e).__name__}: {e}")
        try:
            await ws.close(code=1011, reason="Internal error")  # [S4] No detail leak
        except Exception:
            pass


# ────────────────────────────────────────────────────────
# Face Verification Endpoints
# ────────────────────────────────────────────────────────
@app.post(
    "/api/face/enroll", 
    response_model=FaceResponse,
    responses={
        400: {"description": "Format input/gambar Invalid"},
        404: {"description": "NIM tidak ditemukan di DB"},
        409: {"description": "Wajah sudah terdaftar"},
        422: {"description": "Gagal deteksi wajah (misal tak ada wajah di kamera)"},
    }
)
async def api_face_enroll(req: FaceRequest):
    """
    Pendaftaran wajah mahasiswa (pertama kali).
    Menerima foto wajah Base64 + NIM + Nama → simpan 512-D embedding ke DB.
    """
    nim = _validate_nim(req.nim)

    # Validasi: NIM harus ada di DB & (opsional) Nama sesuai dengan KTM
    db_res = verify_student(nim, req.nama)
    if not db_res.get("verified"):
        raise HTTPException(status_code=404, detail=db_res.get("message", "NIM tidak ditemukan / Nama tidak cocok"))

    # Validasi: belum punya encoding (jika sudah ada, pakai /verify)
    existing = load_face_encoding(nim)
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Wajah untuk {nim} sudah terdaftar. Gunakan /api/face/verify."
        )

    # Decode gambar
    frame = decode_base64_image(req.image_base64)
    if frame is None:
        raise HTTPException(status_code=400, detail="Gagal decode gambar. Pastikan format Base64 valid.")

    # Enroll: extract 512-D embedding (Aman dari Race Condition)
    t_start = time.perf_counter()
    async with face_verifier_lock:
        fv = _get_face_verifier()
        enroll_res = await asyncio.to_thread(fv.enroll, frame)
    elapsed = round((time.perf_counter() - t_start) * 1000, 1)

    if not enroll_res["success"]:
        return JSONResponse(status_code=422, content={
            "status": "failed",
            "message": enroll_res["message"],
            "processing_time_ms": elapsed,
        })

    # Simpan encoding ke DB + check-in
    save_face_encoding(nim, enroll_res["encoding"])
    ci = check_in(nim)

    return FaceResponse(
        status="enrolled",
        nim=nim,
        nama=db_res.get("nama_db"),
        checkin=ci,
        processing_time_ms=elapsed,
    )


@app.post(
    "/api/face/verify", 
    response_model=FaceResponse,
    responses={
        400: {"description": "Format input/gambar Invalid"},
        403: {"description": "Wajah TIDAK cocok dengan DB"},
        404: {"description": "Wajah/NIM belum terdaftar"},
    }
)
async def api_face_verify(req: FaceRequest):
    """
    Verifikasi wajah mahasiswa (absensi harian).
    Menerima foto wajah Base64 + NIM → compare dengan DB → check-in jika cocok.
    """
    nim = _validate_nim(req.nim)

    # Validasi: NIM harus ada di DB
    db_res = verify_student(nim, req.nama)
    if not db_res.get("verified"):
        raise HTTPException(status_code=404, detail=db_res.get("message", "NIM tidak ditemukan / Nama tidak cocok"))

    # Validasi: harus sudah punya encoding
    db_encoding = load_face_encoding(nim)
    if db_encoding is None:
        raise HTTPException(
            status_code=404,
            detail=f"Wajah untuk {nim} belum terdaftar. Gunakan /api/face/enroll dulu."
        )

    # Decode gambar
    frame = decode_base64_image(req.image_base64)
    if frame is None:
        raise HTTPException(status_code=400, detail="Gagal decode gambar. Pastikan format Base64 valid.")

    # Verify: compare live face vs DB encoding (Aman dari Race Condition)
    t_start = time.perf_counter()
    async with face_verifier_lock:
        fv = _get_face_verifier()
        fv.set_reference_from_encoding(db_encoding)
        verify_res = await asyncio.to_thread(fv.verify, frame)
        fv.clear_reference()  # Bersihkan state setelah selesai
    elapsed = round((time.perf_counter() - t_start) * 1000, 1)

    if verify_res["verified"]:
        ci = check_in(nim)
        return FaceResponse(
            status="verified",
            nim=nim,
            nama=db_res.get("nama_db"),
            similarity=verify_res["similarity"],
            checkin=ci,
            processing_time_ms=elapsed,
        )

    return JSONResponse(status_code=403, content={
        "status": "rejected",
        "nim": nim,
        "similarity": verify_res["similarity"],
        "message": verify_res["message"],
        "processing_time_ms": elapsed,
    })

@app.delete(
    "/api/face/{nim}",
    dependencies=[Depends(require_admin_key)],
    responses={
        401: {"description": "Unauthorized — Header X-Admin-Key salah/tidak ada"},
        404: {"description": "NIM tidak ditemukan"},
    }
)
async def api_face_reset(nim: str):
    """
    [ADMIN] Reset / hapus face encoding mahasiswa dari database.
    Berguna jika mahasiswa salah melakukan enrollment wajah.
    Membutuhkan: Header `X-Admin-Key: <ADMIN_API_KEY>`
    """
    nim = _validate_nim(nim)
    res = delete_face_encoding(nim)
    if not res["success"]:
        raise HTTPException(status_code=404, detail=res["message"])
    return {"status": "success", "message": res["message"]}

# ────────────────────────────────────────────────────────
# Checkout & Status Endpoints
# ────────────────────────────────────────────────────────
@app.post("/api/checkout/{nim}")
async def api_checkout(nim: str):
    """Check-out mahasiswa dari lab. [S2] NIM validated."""
    nim = _validate_nim(nim)
    return check_out(nim)


@app.get(
    "/api/status",
    dependencies=[Depends(require_admin_key)]
)
async def api_status():
    """[ADMIN] List peminjaman aktif. Membutuhkan X-Admin-Key header."""
    active = get_active_peminjaman()
    return {
        "active_count": len(active),
        "peminjaman": active,
    }


# ────────────────────────────────────────────────────────
# Run (development only)
# ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
