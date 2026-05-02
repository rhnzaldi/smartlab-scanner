import time
import asyncio
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from db.database import verify_student, load_face_encoding, save_face_encoding, delete_face_encoding, check_in
from core.dependencies import get_current_student, get_current_admin, get_face_verifier, get_face_lock
from core.utils import validate_nim
from routers.scan import decode_base64_image

router = APIRouter(tags=["Face"])

class FaceRequest(BaseModel):
    nim: str
    nama: Optional[str] = None
    image_base64: str

class FaceResponse(BaseModel):
    status: str
    nim: Optional[str] = None
    nama: Optional[str] = None
    similarity: Optional[float] = None
    checkin: Optional[dict] = None
    processing_time_ms: Optional[float] = None
    message: Optional[str] = None

@router.post(
    "/face/enroll", 
    response_model=FaceResponse,
    responses={
        400: {"description": "Format input/gambar Invalid"},
        404: {"description": "NIM tidak ditemukan di DB"},
        409: {"description": "Wajah sudah terdaftar"},
        422: {"description": "Gagal deteksi wajah (misal tak ada wajah di kamera)"},
    }
)
async def api_face_enroll(req: FaceRequest, current_user: dict = Depends(get_current_student)):
    logged_nim = current_user["username"]
    req_nim = validate_nim(req.nim)
    if req_nim != logged_nim:
        raise HTTPException(status_code=403, detail="NIM harus sesuai dengan akun mahasiswa yang login.")
    nim = req_nim

    db_res = await asyncio.to_thread(verify_student, nim, req.nama)
    if not db_res.get("verified"):
        raise HTTPException(status_code=404, detail=db_res.get("message", "NIM tidak ditemukan / Nama tidak cocok"))

    existing = await asyncio.to_thread(load_face_encoding, nim)
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Wajah untuk {nim} sudah terdaftar. Gunakan /api/face/verify."
        )

    frame = decode_base64_image(req.image_base64)
    if frame is None:
        raise HTTPException(status_code=400, detail="Gagal decode gambar. Pastikan format Base64 valid.")

    t_start = time.perf_counter()
    # [P-04] Pakai per-NIM lock: mahasiswa lain tidak perlu antri
    nim_lock = await get_face_lock(nim)
    async with nim_lock:
        fv = get_face_verifier()
        enroll_res = await asyncio.to_thread(fv.enroll, frame)
    elapsed = round((time.perf_counter() - t_start) * 1000, 1)

    if not enroll_res["success"]:
        if enroll_res.get("spoof_detected"):
            raise HTTPException(
                status_code=406,
                detail=enroll_res.get("message", "Akses Ditolak: Indikasi Wajah Palsu/Spoofing.")
            )
            
        return JSONResponse(status_code=422, content={
            "status": "failed",
            "message": enroll_res["message"],
            "processing_time_ms": elapsed,
        })

    await asyncio.to_thread(save_face_encoding, nim, enroll_res["encoding"])
    ci = await asyncio.to_thread(check_in, nim)

    return FaceResponse(
        status="enrolled",
        nim=nim,
        nama=db_res.get("nama_db"),
        checkin=ci,
        processing_time_ms=elapsed,
    )


@router.post(
    "/face/verify", 
    response_model=FaceResponse,
    responses={
        400: {"description": "Format input/gambar Invalid"},
        403: {"description": "Wajah TIDAK cocok dengan DB"},
        404: {"description": "Wajah/NIM belum terdaftar"},
        406: {"description": "Terdeteksi Indikasi Spoofing (Wajah Palsu / Layar HP)"},
    }
)
async def api_face_verify(req: FaceRequest, current_user: dict = Depends(get_current_student)):
    logged_nim = current_user["username"]
    req_nim = validate_nim(req.nim)
    if req_nim != logged_nim:
        raise HTTPException(status_code=403, detail="NIM harus sesuai dengan akun mahasiswa yang login.")
    nim = req_nim

    db_res = await asyncio.to_thread(verify_student, nim, req.nama)
    if not db_res.get("verified"):
        raise HTTPException(status_code=404, detail=db_res.get("message", "NIM tidak ditemukan / Nama tidak cocok"))

    db_encoding = await asyncio.to_thread(load_face_encoding, nim)
    if db_encoding is None:
        raise HTTPException(
            status_code=404,
            detail=f"Wajah untuk {nim} belum terdaftar. Gunakan /api/face/enroll dulu."
        )

    frame = decode_base64_image(req.image_base64)
    if frame is None:
        raise HTTPException(status_code=400, detail="Gagal decode gambar. Pastikan format Base64 valid.")

    t_start = time.perf_counter()
    # [P-04] Pakai per-NIM lock: mahasiswa lain tidak perlu antri
    nim_lock = await get_face_lock(nim)
    async with nim_lock:
        fv = get_face_verifier()
        fv.set_reference_from_encoding(db_encoding)
        verify_res = await asyncio.to_thread(fv.verify, frame)
        fv.clear_reference()
    elapsed = round((time.perf_counter() - t_start) * 1000, 1)

    if verify_res.get("spoof_detected"):
        raise HTTPException(
            status_code=406,
            detail=verify_res.get("message", "Akses Ditolak: Indikasi Wajah Palsu/Spoofing.")
        )

    if verify_res["verified"]:
        ci = await asyncio.to_thread(check_in, nim)
        
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


@router.delete(
    "/face/{nim}",
    dependencies=[Depends(get_current_admin)],
    responses={
        401: {"description": "Unauthorized — Token JWT tidak valid/kedaluwarsa"},
        404: {"description": "NIM tidak ditemukan"},
    }
)
async def api_face_reset(nim: str):
    nim = validate_nim(nim)
    res = await asyncio.to_thread(delete_face_encoding, nim)
    if not res["success"]:
        raise HTTPException(status_code=404, detail=res["message"])
    return {"status": "success", "message": res["message"]}
