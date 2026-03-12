"""
Smart-Lab SV IPB — Security Module
Menangani hashing password dan pembuatan/verifikasi JWT token.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

logger = logging.getLogger(__name__)

# ─── Config dari env ────────────────────────────────────
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "")
ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "120"))

# [S-01] Startup security check — jangan izinkan default secret key
_DEFAULT_INSECURE_KEY = "change-me-use-openssl-rand-hex-32"
if not SECRET_KEY or SECRET_KEY == _DEFAULT_INSECURE_KEY:
    # Di production (ditandai via ENVIRONMENT=production), ini hard-fail
    _env = os.environ.get("ENVIRONMENT", "development").lower()
    if _env == "production":
        raise RuntimeError(
            "❌ FATAL: JWT_SECRET_KEY tidak diset atau masih default di environment production. "
            "Generate via: openssl rand -hex 32"
        )
    else:
        # Development: pakai fallback dan beri warning keras
        SECRET_KEY = _DEFAULT_INSECURE_KEY
        logger.warning(
            "⚠️ [S-01] JWT_SECRET_KEY tidak diset! Menggunakan default INSECURE key. "
            "Set JWT_SECRET_KEY di .env sebelum deploy ke production. "
            "Generate via: openssl rand -hex 32"
        )

# ─── Password Hashing ───────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain: str, hashed: str) -> bool:
    """Cocokkan plain password dengan hash yang tersimpan di DB."""
    return pwd_context.verify(plain, hashed)


def get_password_hash(password: str) -> str:
    """Hash password sebelum disimpan ke DB."""
    return pwd_context.hash(password)


# ─── JWT Token ──────────────────────────────────────────
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Buat JWT access token.
    Args:
        data: payload dict, biasanya {"sub": username}
        expires_delta: durasi kedaluwarsa (default dari env var)
    Returns:
        JWT token string
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """
    Dekode dan verifikasi JWT token.
    Raises JWTError jika token tidak valid atau kedaluwarsa.
    Returns:
        Payload dict dari token
    """
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
