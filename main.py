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

# Load .env file sebelum import db/ (agar DB_HOST, DB_USER, dll tersedia)
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError

from ml.pipeline import KTMPipeline, ScanResult
from ml.face_verify import FaceVerifier
from db.database import (
    init_db, verify_student, check_in, check_out,
    get_active_peminjaman, reset_all_peminjaman, NIM_PATTERN,
    save_face_encoding, load_face_encoding, delete_face_encoding,
    get_admin_by_username,
)
from security import verify_password, create_access_token, decode_token

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

# ─── JWT Auth (menggantikan ADMIN_API_KEY) ───────────────
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


async def get_current_admin(token: str = Depends(oauth2_scheme)):
    """
    JWT Dependency Guard untuk endpoint admin.
    Verifikasi Bearer token dari header Authorization.
    """
    try:
        payload = decode_token(token)
        username: str = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Token tidak valid.")
    except JWTError:
        raise HTTPException(status_code=401, detail="Token tidak valid atau kedaluwarsa.")

    admin = get_admin_by_username(username)
    if not admin or not admin.get("is_active"):
        raise HTTPException(status_code=401, detail="Akun admin tidak aktif atau tidak ditemukan.")
    return admin


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
        # Import lazy to avoid circular dependency since database is imported above globally 
        from db.database import has_active_peminjaman
        
        db_res = verify_student(result.nim_final, result.nama)
        response["db_verified"] = db_res["verified"]
        response["db_nama"] = db_res.get("nama_db")
        response["db_prodi"] = db_res.get("prodi")
        response["db_message"] = db_res.get("message")

        if db_res["verified"]:
            # Cek dulu apakah mahasiswa ini sudah/sedang di dalam lab (punya sesi aktif)
            if has_active_peminjaman(result.nim_final):
                response["action_required"] = "already_checked_in"
                nama_panggilan = response.get('db_nama', result.nim_final)
                response["db_message"] = f"Mahasiswa atas nama {nama_panggilan} ({result.nim_final}) masih memiliki sesi di dalam lab. Silakan check-out terlebih dahulu."
            else:
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
    db_res = await asyncio.to_thread(verify_student, nim, req.nama)
    if not db_res.get("verified"):
        raise HTTPException(status_code=404, detail=db_res.get("message", "NIM tidak ditemukan / Nama tidak cocok"))

    # Validasi: belum punya encoding (jika sudah ada, pakai /verify)
    existing = await asyncio.to_thread(load_face_encoding, nim)
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

    # Simpan encoding ke DB + check-in (Aman dari Race Condition)
    await asyncio.to_thread(save_face_encoding, nim, enroll_res["encoding"])
    ci = await asyncio.to_thread(check_in, nim)

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
    db_res = await asyncio.to_thread(verify_student, nim, req.nama)
    if not db_res.get("verified"):
        raise HTTPException(status_code=404, detail=db_res.get("message", "NIM tidak ditemukan / Nama tidak cocok"))

    # Validasi: harus sudah punya encoding
    db_encoding = await asyncio.to_thread(load_face_encoding, nim)
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
        ci = await asyncio.to_thread(check_in, nim)
        
        # Block: Jika mahasiswa sudah punya sesi aktif, tolak verifikasi lagi
        if not ci.get("success"):
            raise HTTPException(
                status_code=409,
                detail=ci.get("message", "Mahasiswa sedang berada di dalam lab.")
            )

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
    dependencies=[Depends(get_current_admin)],
    responses={
        401: {"description": "Unauthorized — Token JWT tidak valid/kedaluwarsa"},
        404: {"description": "NIM tidak ditemukan"},
    }
)
async def api_face_reset(nim: str):
    """
    [ADMIN] Reset / hapus face encoding mahasiswa dari database.
    Berguna jika mahasiswa salah melakukan enrollment wajah.
    Membutuhkan: Bearer Token JWT dari /api/auth/login
    """
    nim = _validate_nim(nim)
    res = await asyncio.to_thread(delete_face_encoding, nim)
    if not res["success"]:
        raise HTTPException(status_code=404, detail=res["message"])
    return {"status": "success", "message": res["message"]}

# ────────────────────────────────────────────────────────
# Checkout, Status, & ACC Endpoints
# ────────────────────────────────────────────────────────
# ─── Login & Status Endpoints ────────────────────────────

@app.post("/api/auth/login")
async def api_login(form: OAuth2PasswordRequestForm = Depends()):
    """
    Login admin. Return JWT access token.
    Format input: form-data (username + password)
    """
    admin = get_admin_by_username(form.username)
    if not admin or not verify_password(form.password, admin["password_hash"]):
        raise HTTPException(
            status_code=401,
            detail="Username atau password salah.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not admin.get("is_active"):
        raise HTTPException(status_code=403, detail="Akun admin tidak aktif.")

    token = create_access_token(data={"sub": admin["username"]})
    return {"access_token": token, "token_type": "bearer"}


@app.post("/api/checkout/{nim}", dependencies=[Depends(get_current_admin)])
async def api_checkout(nim: str):
    """[ADMIN] Check-out mahasiswa dari lab. Membutuhkan Bearer Token."""
    nim = _validate_nim(nim)
    return await asyncio.to_thread(check_out, nim)

@app.post("/api/peminjaman/{pid}/approve", dependencies=[Depends(get_current_admin)])
async def api_approve(pid: int):
    """[ADMIN] Setujui mahasiswa masuk ke dalam lab (ubah status menunggu -> aktif)"""
    from db.database import approve_peminjaman
    res = await asyncio.to_thread(approve_peminjaman, pid)
    if not res["success"]:
        raise HTTPException(status_code=400, detail=res["message"])
    return res

@app.post("/api/peminjaman/{pid}/reject", dependencies=[Depends(get_current_admin)])
async def api_reject(pid: int):
    """[ADMIN] Tolak mahasiswa masuk ke dalam lab (ubah status menunggu -> ditolak)"""
    from db.database import reject_peminjaman
    res = await asyncio.to_thread(reject_peminjaman, pid)
    if not res["success"]:
        raise HTTPException(status_code=400, detail=res["message"])
    return res


@app.get(
    "/api/status",
    dependencies=[Depends(get_current_admin)]
)
async def api_status():
    """[ADMIN] List peminjaman aktif & menunggu ACC. Membutuhkan Bearer Token JWT."""
    peminjaman_list = await asyncio.to_thread(get_active_peminjaman)
    
    # Split them for frontend convenience
    active = [p for p in peminjaman_list if p["status"] == "aktif"]
    pending = [p for p in peminjaman_list if p["status"] == "menunggu"]
    
    return {
        "active_count": len(active),
        "pending_count": len(pending),
        "peminjaman": active,
        "peminjaman_pending": pending
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
