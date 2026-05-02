import base64
import json
import logging
import time
import asyncio
import numpy as np
import cv2
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, File, UploadFile, HTTPException, Depends, Query
from jose import JWTError

from security import decode_token
from ml.pipeline import ScanResult
from db.database import verify_student, has_active_peminjaman, has_face_encoding, get_user_by_username
from core.dependencies import get_pipeline

logger = logging.getLogger("smartlab-api")
router = APIRouter(tags=["Scan"])

MAX_UPLOAD_SIZE_MB = 10
MAX_UPLOAD_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
WS_MIN_INTERVAL_MS = 300

def decode_base64_image(data: str):
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
    if result.success:
        db_res = verify_student(result.nim_final, result.nama)
        response["db_verified"] = db_res["verified"]
        response["db_nama"] = db_res.get("nama_db")
        response["db_prodi"] = db_res.get("prodi")
        response["db_message"] = db_res.get("message")

        if db_res["verified"]:
            if has_active_peminjaman(result.nim_final):
                response["action_required"] = "already_checked_in"
                nama_panggilan = response.get('db_nama', result.nim_final)
                response["db_message"] = (f"Mahasiswa atas nama {nama_panggilan} ({result.nim_final}) "
                    "masih memiliki sesi di dalam lab. Silakan check-out terlebih dahulu.")
            else:
                response["action_required"] = (
                    "face_verify" if has_face_encoding(result.nim_final) else "face_enroll"
                )
    else:
        response["db_verified"] = False
        response["action_required"] = None
        if result.status == "qr_missing":
            response["db_message"] = "QR Code tidak terdeteksi. Pastikan KTM memiliki QR code yang terlihat."
        elif result.status == "nim_mismatch":
            response["db_message"] = (
                f"NIM tidak cocok: QR='{result.nim_qr}' vs OCR='{result.nim_ocr}'. "
                "Pastikan KTM asli dan tidak tertutup."
            )
        elif result.status == "face_photo_missing":
            response["db_message"] = "Foto wajah pada KTM tidak terdeteksi. Pastikan KTM terlihat jelas."
        elif result.status == "incomplete":
            response["db_message"] = "Data KTM tidak lengkap. Pastikan NIM, Nama, QR, dan foto terlihat."
    return response


from core.dependencies import get_pipeline, get_current_student

@router.post("/api/scan")
async def scan_image(file: UploadFile = File(...), current_user: dict = Depends(get_current_student)):
    pipeline = get_pipeline()
    if not pipeline or not pipeline.is_ready():
        raise HTTPException(status_code=503, detail="ML pipeline not ready.")

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
    response["processing_time_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
    return response


@router.websocket("/ws/scan")
async def websocket_scan(ws: WebSocket, token: str = Query(None)):
    if not token:
        await ws.close(code=1008, reason="Token required")
        return
        
    try:
        payload = decode_token(token)
        username = payload.get("sub")
        if not username:
            await ws.close(code=1008, reason="Invalid token")
            return
            
        user = get_user_by_username(username)
        if not user or not user.get("is_active") or user.get("role") != "mahasiswa":
            await ws.close(code=1008, reason="Unauthorized")
            return
    except JWTError:
        await ws.close(code=1008, reason="Token expired or invalid")
        return

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
            except Exception as e:
                logger.error(f"WS parse error: {e}")
                image_data = raw_data

            frame = decode_base64_image(image_data)
            if frame is None:
                continue

            pipeline = get_pipeline()
            if pipeline and pipeline.is_ready():
                result = await asyncio.to_thread(pipeline.process_frame, frame)
                response = _process_scan_result(result)
            else:
                response = {"status": "error", "message": "Model not loaded"}

            await ws.send_json(response)
            last_process_time = time.time() * 1000
    except WebSocketDisconnect:
        logger.info(f"🔌 Disconnected: {client_id}")
    except Exception as e:
        logger.error(f"WS Error: {e}")

