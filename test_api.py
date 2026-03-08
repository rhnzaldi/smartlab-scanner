"""
Smart-Lab SV IPB — API Integration Tests
Menguji FastAPI endpoints tanpa perlu menjalankan server secara manual.

Requirements:
    pip install pytest httpx
    
Usage:
    pytest test_api.py -v
"""

import base64
import os
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from main import app
from db.database import init_db, reset_all_peminjaman, get_connection

# Setup TestClient (bypass jaringan, langsung test ASGI routes)
client = TestClient(app)

# Konstanta Test
TEST_NIM = "J0403231061"
TEST_NAME = "Raihan Zaldi"

# ────────────────────────────────────────────────────────
# Test Fixtures & Dummies
# ────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def setup_teardown():
    """Jalankan sebelum dan sesudah setiap test"""
    # Pastikan DB init
    init_db()
    # Bersihkan state peminjaman
    reset_all_peminjaman()
    
    # Hapus test face encoding jika ada
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("UPDATE mahasiswa SET face_encoding = NULL WHERE nim = %s", (TEST_NIM,))
        
    yield
    
    # Teardown
    reset_all_peminjaman()


def get_dummy_face_base64() -> str:
    """Buat gambar wajah dummy hitam dengan ukuran 640x640."""
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    # Gambar kotak / bentuk di dalamnya agar file valid
    cv2.rectangle(img, (200, 200), (440, 440), (255, 255, 255), -1)
    
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')


# ────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────

def get_admin_token() -> dict:
    """Helper untuk men-generate bearer token via login route testclient."""
    login_data = {
        "username": "admin",
        "password": "admin123"
    }
    res = client.post("/api/auth/login", data=login_data)
    if res.status_code == 200:
        return {"Authorization": f"Bearer {res.json()['access_token']}"}
    return {}

def test_health_check():
    """Test health check selalu jalan."""
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_api_status_empty():
    """Test get status waktu lab kosong (requires admin key)."""
    headers = get_admin_token()
    res = client.get("/api/status", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["active_count"] == 0


def test_api_status_unauthorized():
    """Test status ditolak tanpa admin key."""
    res = client.get("/api/status")
    assert res.status_code == 401


def test_api_checkout_not_found():
    """Test checkout NIM yang belum masuk."""
    headers = get_admin_token()
    res = client.post(f"/api/checkout/{TEST_NIM}", headers=headers)
    assert res.status_code == 200
    assert res.json()["success"] == False
    assert "tidak memiliki peminjaman aktif" in res.json()["message"]


def test_api_face_enroll_invalid_base64():
    """Test bad image data ditolak oleh face enroll."""
    data = {
        "nim": TEST_NIM,
        "nama": TEST_NAME,
        "image_base64": "invalid_base64_string"
    }
    res = client.post("/api/face/enroll", json=data)
    assert res.status_code == 400
    assert "Gagal decode gambar" in res.json()["detail"]


def test_api_face_enroll_no_face_detected():
    """
    Test enroll dengan dummy image (tidak ada wajah manusia).
    FaceVerifier harus return 422 karena tidak menemukan wajah.
    """
    data = {
        "nim": TEST_NIM,
        "nama": TEST_NAME,
        "image_base64": get_dummy_face_base64()
    }
    res = client.post("/api/face/enroll", json=data)
    assert res.status_code == 422
    assert "Wajah tidak terdeteksi" in res.json()["message"]


def test_api_face_reset():
    """Test admin reset endpoint berfungsi dengan api key."""
    headers = get_admin_token()
    res = client.delete(f"/api/face/{TEST_NIM}", headers=headers)
    assert res.status_code == 200

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT face_encoding FROM mahasiswa WHERE nim = %s", (TEST_NIM,))
            row = cursor.fetchone()
        assert row["face_encoding"] is None


def test_api_face_reset_unauthorized():
    """Test reset ditolak tanpa admin key."""
    res = client.delete(f"/api/face/{TEST_NIM}")
    assert res.status_code == 401

