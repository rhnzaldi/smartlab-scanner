import asyncio
import logging
from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError

from security import decode_token
from db.database import get_user_by_username
from ml.pipeline import KTMPipeline

logger = logging.getLogger("smartlab-api")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

async def get_current_admin(token: str = Depends(oauth2_scheme)):
    """
    JWT Dependency Guard untuk endpoint admin.
    Verifikasi Bearer token dari header Authorization.
    """
    try:
        payload = decode_token(token)
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if not username or role != "admin":
            raise HTTPException(status_code=401, detail="Token tidak valid.")
    except JWTError:
        raise HTTPException(status_code=401, detail="Token tidak valid atau kedaluwarsa.")

    admin = get_user_by_username(username)
    if not admin or admin.get("role") != "admin" or not admin.get("is_active"):
        raise HTTPException(status_code=401, detail="Akun admin tidak aktif atau tidak ditemukan.")
    return admin


async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    JWT Dependency Guard untuk semua login user.
    Verifikasi Bearer token dan pastikan akun aktif.
    """
    try:
        payload = decode_token(token)
        username: str = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Token tidak valid.")
    except JWTError:
        raise HTTPException(status_code=401, detail="Token tidak valid atau kedaluwarsa.")

    user = get_user_by_username(username)
    if not user or not user.get("is_active"):
        raise HTTPException(status_code=401, detail="Akun tidak aktif atau tidak ditemukan.")
    return user


async def get_current_student(current_user: dict = Depends(get_current_user)):
    """
    JWT Dependency Guard khusus mahasiswa.
    """
    if current_user.get("role") != "mahasiswa":
        raise HTTPException(status_code=403, detail="Akses mahasiswa diperlukan.")
    return current_user


# ────────────────────────────────────────────────────────
# ML Engine States (Global)
# ────────────────────────────────────────────────────────
pipeline: Optional[KTMPipeline] = None

# [P-04] Per-NIM locking: mahasiswa berbeda bisa face-verify paralel.
# Hanya NIM yang sama yang antri — mencegah race condition tanpa bottleneck global.
_face_locks: dict[str, asyncio.Lock] = {}
_face_locks_meta_lock = asyncio.Lock()  # Guard untuk akses _face_locks dict

async def get_face_lock(nim: str) -> asyncio.Lock:
    """Return (atau buat) asyncio.Lock per-NIM."""
    async with _face_locks_meta_lock:
        if nim not in _face_locks:
            _face_locks[nim] = asyncio.Lock()
        return _face_locks[nim]

_face_verifier_instance = None


def set_pipeline(p: KTMPipeline):
    global pipeline
    pipeline = p


def get_pipeline():
    global pipeline
    return pipeline


def get_face_verifier():
    global _face_verifier_instance
    if _face_verifier_instance is None:
        try:
            from ml.face_verify import FaceVerifier
            _face_verifier_instance = FaceVerifier()
        except Exception as e:
            logger.warning(f"⚠️ FaceVerifier tidak dapat dimuat: {e}")
            raise HTTPException(status_code=503, detail="Face verifier tidak tersedia.")
    return _face_verifier_instance
