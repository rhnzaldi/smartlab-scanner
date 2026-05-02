import time
import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordRequestForm
from security import create_access_token, verify_password
from db.database import get_user_by_username
from core.dependencies import get_current_admin

logger = logging.getLogger("smartlab-api")
router = APIRouter(tags=["Auth"])

_LOGIN_MAX_ATTEMPTS = 5
_LOGIN_WINDOW_SECONDS = 300
# [S-03] Dual tracking: per-IP DAN per-username
# Attacker yang ganti IP tapi tetap pakai username yang sama tetap terkena limit.
# Catatan: In-memory — reset saat server restart. Untuk multi-process/production,
# ganti dengan Redis: pip install redis, lalu pakai redis.Redis.incr() + EXPIRE.
_LOGIN_ATTEMPTS_IP: dict = {}       # {ip: [timestamp, ...]}
_LOGIN_ATTEMPTS_USER: dict = {}     # {username: [timestamp, ...]}


def _check_rate_limit(key: str, store: dict, label: str) -> int:
    """
    Cek dan catat attempt untuk key tertentu.
    Return: sisa attempt yang dibolehkan. 0 = terkena limit.
    """
    now_ts = time.time()
    attempts = store.get(key, [])
    # Hapus attempt di luar window
    store[key] = [t for t in attempts if now_ts - t < _LOGIN_WINDOW_SECONDS]
    remaining = _LOGIN_MAX_ATTEMPTS - len(store[key])
    if remaining <= 0:
        logger.warning(f"[S-03] Rate limit tercapai untuk {label}: {key}")
        return 0
    store[key].append(now_ts)
    return remaining - 1  # -1 karena attempt ini baru ditambahkan


@router.post("/auth/login")
async def api_login(
    request: Request,
    form: OAuth2PasswordRequestForm = Depends(),
):
    """
    Login admin atau mahasiswa. Return JWT access token dan role.
    """
    client_ip = request.client.host if request.client else "unknown"

    # [S-03] Cek limit per-IP
    remaining_ip = _check_rate_limit(client_ip, _LOGIN_ATTEMPTS_IP, "IP")
    if remaining_ip == 0:
        raise HTTPException(
            status_code=429,
            detail=f"Terlalu banyak percobaan login dari IP ini. Coba lagi dalam {_LOGIN_WINDOW_SECONDS} detik.",
            headers={
                "Retry-After": str(_LOGIN_WINDOW_SECONDS),
                "X-RateLimit-Remaining": "0",
            },
        )

    # [S-03] Cek limit per-username (mencegah credential stuffing ganti IP)
    remaining_user = _check_rate_limit(form.username, _LOGIN_ATTEMPTS_USER, "username")
    if remaining_user == 0:
        raise HTTPException(
            status_code=429,
            detail=f"Terlalu banyak percobaan login untuk akun ini. Coba lagi dalam {_LOGIN_WINDOW_SECONDS} detik.",
            headers={
                "Retry-After": str(_LOGIN_WINDOW_SECONDS),
                "X-RateLimit-Remaining": "0",
            },
        )

    user = get_user_by_username(form.username)
    if user and user.get("role") == "mahasiswa":
        password_to_verify = form.password.lower()
    else:
        password_to_verify = form.password

    if not user or not user.get("password_hash") or not verify_password(password_to_verify, user["password_hash"]):
        logger.info(f"[S-03] Gagal login: username='{form.username}' ip={client_ip}")
        raise HTTPException(
            status_code=401,
            detail="Username atau password salah.",
            headers={
                "WWW-Authenticate": "Bearer",
                "X-RateLimit-Remaining": str(min(remaining_ip, remaining_user)),
            },
        )
    if not user.get("is_active"):
        raise HTTPException(status_code=403, detail="Akun tidak aktif.")

    # Login berhasil → reset attempt counter untuk IP dan username ini
    _LOGIN_ATTEMPTS_IP.pop(client_ip, None)
    _LOGIN_ATTEMPTS_USER.pop(form.username, None)

    token = create_access_token(data={"sub": user["username"], "role": user["role"]})
    redirect_url = "/adminlab/dashboard" if user["role"] == "admin" else "/"
    logger.info(f"✅ Login berhasil: '{form.username}' (role={user['role']}) dari {client_ip}")

    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user["role"],
        "name": user.get("nama") or user.get("username"),
        "redirect_url": redirect_url,
    }



@router.get("/auth/me")
async def api_verify_token(admin: dict = Depends(get_current_admin)):
    """
    [S-05] Validasi token JWT admin ke backend.
    Frontend admin layout memanggil ini saat mount untuk memastikan token valid
    dan belum expired — bukan sekadar mengecek keberadaan key di localStorage.
    """
    return {
        "valid": True,
        "username": admin.get("username"),
        "role": admin.get("role"),
        "name": admin.get("nama") or admin.get("username"),
    }
