"""
Smart-Lab SV IPB — Security Module
Menangani hashing password dan pembuatan/verifikasi JWT token.
"""

import os
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

# ─── Config dari env ────────────────────────────────────
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "change-me-use-openssl-rand-hex-32")
ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "120"))

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
