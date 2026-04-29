"""
Smart-Lab SV IPB — FastAPI Backend
REST API + WebSocket endpoint untuk KTM scanning.
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
# --- PENAMBAHAN IMPORT ---
from pydantic import BaseModel

from ml.pipeline import KTMPipeline, ScanResult
from db.database import (
    init_db, verify_student, check_in, check_out,
    get_active_peminjaman, reset_all_peminjaman, NIM_PATTERN,
)

# ────────────────────────────────────────────────────────
# Constants [M1]
# ────────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB = 10
MAX_UPLOAD_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
WS_MIN_INTERVAL_MS = 300

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

# --- PENAMBAHAN SCHEMA LOGIN ---


class LoginRequest(BaseModel):
    username: str
    password: str


# ────────────────────────────────────────────────────────
# Global pipeline (singleton)
# ────────────────────────────────────────────────────────
pipeline: Optional[KTMPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("🚀 Starting Smart-Lab ML Engine...")
    init_db()
    logger.info("✅ Database ready")
    pipeline = KTMPipeline(model_path="models/best.pt")
    if pipeline.is_ready():
        logger.info("✅ Pipeline ready!")
    else:
        logger.warning("⚠️ Pipeline loaded but model NOT FOUND.")
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────────────────────────────────────────────
# AUTH ENDPOINT (NEWly Added)
# ────────────────────────────────────────────────────────


@app.post("/auth/login")
async def login(data: LoginRequest):
    """
    Endpoint Login untuk Admin & Mahasiswa.
    """
    logger.info(f"🔑 Login attempt for: {data.username}")

    # Logika Login Admin
    if data.username == "admin" and data.password == "admin123":
        return {
            "access_token": "secret-token-admin",
            "role": "admin",
            "name": "Administrator Lab"
        }

    # Logika Login Mahasiswa (Berdasarkan format NIM)
    elif data.username.startswith("J") or data.username.startswith("j"):
        # Anda bisa menambahkan verify_student(data.username, ...) di sini jika perlu
        return {
            "access_token": "token-mahasiswa",
            "role": "mahasiswa",
            "name": data.username
        }

    raise HTTPException(
        status_code=401, detail="Username atau Password salah.")


# ────────────────────────────────────────────────────────
# Input Validation Helpers
# ────────────────────────────────────────────────────────
_NIM_RE = re.compile(NIM_PATTERN)


def _validate_nim(nim: str) -> str:
    nim = nim.strip().upper()
    if not _NIM_RE.match(nim):
        raise HTTPException(
            status_code=400,
            detail=f"Format NIM tidak valid: '{nim}'."
        )
    return nim


def _ensure_pipeline_ready():
    if not pipeline or not pipeline.is_ready():
        raise HTTPException(status_code=503, detail="ML pipeline not ready.")


# ────────────────────────────────────────────────────────
# Helper
# ────────────────────────────────────────────────────────
def decode_base64_image(data: str) -> Optional[np.ndarray]:
    try:
        if "," in data:
            data = data.split(",", 1)[1]
        img_bytes = base64.b64decode(data)
        if len(img_bytes) > MAX_UPLOAD_BYTES:
            return None
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.error(f"Failed to decode image: {type(e).__name__}")
        return None


def _process_scan_result(result: ScanResult) -> dict:
    response = result.to_dict()
    if result.nim_final and result.nama:
        db_res = verify_student(result.nim_final, result.nama)
        response["db_verified"] = db_res["verified"]
        response["db_nama"] = db_res.get("nama_db")
        response["db_prodi"] = db_res.get("prodi")
        response["db_message"] = db_res.get("message")

        if db_res["verified"]:
            response["action_required"] = "face_verification"
    return response


# ────────────────────────────────────────────────────────
# REST Endpoints
# ────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": pipeline.is_ready() if pipeline else False,
    }


@app.post("/api/scan")
async def scan_image(file: UploadFile = File(...)):
    _ensure_pipeline_ready()
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File terlalu besar.")

    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    t_start = time.perf_counter()
    result = pipeline.process_frame(frame)
    response = _process_scan_result(result)
    response["processing_time_ms"] = round(
        (time.perf_counter() - t_start) * 1000, 1)
    return response


@app.websocket("/ws/scan")
async def websocket_scan(ws: WebSocket):
    await ws.accept()
    client_id = id(ws)
    last_process_time = 0.0
    try:
        while True:
            raw_data = await ws.receive_text()
            now = time.time() * 1000
            if now - last_process_time < WS_MIN_INTERVAL_MS:
                continue

            try:
                msg = json.loads(raw_data)
                image_data = msg.get("image", msg.get("frame", raw_data))
            except:
                image_data = raw_data

            frame = decode_base64_image(image_data)
            if frame is None:
                continue

            if pipeline and pipeline.is_ready():
                result = await asyncio.to_thread(pipeline.process_frame, frame)
                response = _process_scan_result(result)
            else:
                response = {"status": "error", "message": "Model not loaded"}

            await ws.send_json(response)
            last_process_time = time.time() * 1000
    except WebSocketDisconnect:
        logger.info(f"🔌 Disconnected: {client_id}")


@app.post("/api/checkout/{nim}")
async def api_checkout(nim: str):
    nim = _validate_nim(nim)
    return check_out(nim)


@app.get("/api/status")
async def api_status():
    active = get_active_peminjaman()
    return {"active_count": len(active), "peminjaman": active}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
