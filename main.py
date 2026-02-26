"""
Smart-Lab SV IPB — FastAPI Backend
REST API + WebSocket endpoint untuk KTM scanning.

NFR Refactored:
- [S1] CORS origins dari environment variable
- [S2] Input validation (NIM format, file size)
- [S3] File size limit (MAX_UPLOAD_SIZE_MB)
- [R3] WebSocket DB integration
- [O1] Request timing in response headers

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import base64
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ml.pipeline import KTMPipeline, ScanResult
from db.database import (
    init_db, verify_student, check_in, check_out,
    get_active_peminjaman, reset_all_peminjaman, NIM_PATTERN,
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
    allow_methods=["GET", "POST"],
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
    Process scan result: verify DB + check-in jika verified.
    [M4] Single source of truth untuk scan→verify→checkin logic.
    
    TODO: API ini BELUM menjalankan face verification.
    Check-in via API langsung tanpa pengecekan wajah.
    Untuk production, tambahkan face verify sebelum check-in.
    """
    response = result.to_dict()

    if result.nim_final and result.nama:
        db_res = verify_student(result.nim_final, result.nama)
        response["db_verified"] = db_res["verified"]
        response["db_nama"] = db_res.get("nama_db")
        response["db_prodi"] = db_res.get("prodi")
        response["db_message"] = db_res.get("message")
        if db_res["verified"]:
            ci_res = check_in(result.nim_final)
            response["checkin"] = ci_res

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
# Checkout & Status Endpoints
# ────────────────────────────────────────────────────────
@app.post("/api/checkout/{nim}")
async def api_checkout(nim: str):
    """Check-out mahasiswa dari lab. [S2] NIM validated."""
    nim = _validate_nim(nim)
    return check_out(nim)


@app.get("/api/status")
async def api_status():
    """List peminjaman aktif."""
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
