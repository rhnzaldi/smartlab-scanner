"""
Smart-Lab SV IPB — Database Module (MySQL)
MySQL database untuk manajemen mahasiswa dan peminjaman lab.

Migrated from SQLite to MySQL (PyMySQL driver).
- [R1] Context manager untuk semua koneksi (no connection leak)
- [R2] Duration fix: total_seconds() bukan .seconds
- [M1] Named constants (no magic numbers)
- [M7] Consistent return types
- [O2] DB operation timing via decorator
"""

import pymysql
from pymysql.cursors import DictCursor
import logging
import os
import functools
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, Dict, List, Any
from difflib import SequenceMatcher

import numpy as np

logger = logging.getLogger(__name__)

from dbutils.pooled_db import PooledDB

# ────────────────────────────────────────────────────────
# Constants [M1]
# ────────────────────────────────────────────────────────
DB_HOST = os.environ.get("DB_HOST", "127.0.0.1")
DB_PORT = int(os.environ.get("DB_PORT", "3306"))
DB_USER = os.environ.get("DB_USER", "root")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_NAME = os.environ.get("DB_NAME", "smartlab_db")

NAME_MATCH_THRESHOLD = 0.5       # minimum similarity untuk fuzzy name match
NIM_PATTERN = r"^J[A-Z0-9]{6,12}$"  # valid NIM format untuk input validation [S2]
DEFAULT_LAB = "Lab Smart-Lab"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


# ────────────────────────────────────────────────────────
# Connection Management (Pooled) [R1] [NFR-R]
# ────────────────────────────────────────────────────────
# Inisialisasi Connection Pool Global menggunakan DBUtils
db_pool = PooledDB(
    creator=pymysql,
    maxconnections=20,     # Max 20 concurrent connections
    mincached=2,           # Keep at least 2 connections alive
    maxcached=5,           # Maksimum 5 idle connections di memory
    blocking=True,         # Jika pool penuh, tunggu sampai ada yang kosong
    ping=1,                # Check koneksi aktif sebelum digunakan
    host=DB_HOST,
    port=DB_PORT,
    user=DB_USER,
    password=DB_PASSWORD,
    database=DB_NAME,
    cursorclass=DictCursor,
    charset="utf8mb4",
    autocommit=False,
)

@contextmanager
def get_connection():
    """
    Context manager untuk MySQL connection via PooledDB.
    Berfungsi mengambil koneksi idle dari memory (sangat cepat).
    Otomatis commit saat sukses, rollback saat error, membebaskan koneksi kembali ke pool.
    """
    conn = db_pool.connection()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"❌ DB Transaction Error: {e}")
        raise
    finally:
        conn.close() # Return connection to pool rather than dropping it



# ────────────────────────────────────────────────────────
# Observability: DB Operation Timing [O2]
# ────────────────────────────────────────────────────────
def _timed_db_op(func):
    """Decorator: log waktu eksekusi DB operation."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t_start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.debug(f"  DB [{func.__name__}] completed in {elapsed_ms:.1f}ms")
        return result
    return wrapper


# ────────────────────────────────────────────────────────
# Schema & Seed
# ────────────────────────────────────────────────────────
def init_db():
    """Create tables if not exist and seed initial data."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            # Tabel mahasiswa
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mahasiswa (
                    nim VARCHAR(20) PRIMARY KEY,
                    nama VARCHAR(100) NOT NULL,
                    prodi VARCHAR(50),
                    angkatan INT,
                    status VARCHAR(20) DEFAULT 'aktif',
                    face_encoding LONGBLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)

            # Tabel peminjaman
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS peminjaman (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    nim VARCHAR(20) NOT NULL,
                    lab VARCHAR(100) NOT NULL DEFAULT 'Lab Smart-Lab',
                    waktu_masuk TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    waktu_keluar TIMESTAMP NULL,
                    status VARCHAR(20) DEFAULT 'aktif',
                    scan_confidence FLOAT,
                    catatan TEXT,
                    FOREIGN KEY (nim) REFERENCES mahasiswa(nim)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)

            # Index untuk query yang sering dipakai [P1]
            try:
                cursor.execute("""
                    CREATE INDEX idx_peminjaman_nim_status
                    ON peminjaman(nim, status)
                """)
            except pymysql.err.OperationalError:
                pass  # index already exists

            # Tabel admin_users
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS admin_users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    email VARCHAR(100) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)

            # Seed default admin
            cursor.execute("SELECT COUNT(*) AS cnt FROM admin_users")
            if cursor.fetchone()["cnt"] == 0:
                # Import di sini untuk hindari circular import
                from security import get_password_hash
                cursor.execute("""
                    INSERT INTO admin_users (username, email, password_hash)
                    VALUES (%s, %s, %s)
                """, (
                    "admin",
                    "admin@smartlab.id",
                    get_password_hash("admin123"),
                ))
                logger.info("✅ Seed admin: username=admin, password=admin123")

            # Seed data mahasiswa
            cursor.execute("SELECT COUNT(*) AS cnt FROM mahasiswa")
            if cursor.fetchone()["cnt"] == 0:
                cursor.execute("""
                    INSERT INTO mahasiswa (nim, nama, prodi, angkatan, status)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    "J0403231061",
                    "Muhammad Raihan Zaldiputra",
                    "TPL",
                    2023,
                    "aktif",
                ))
                logger.info("✅ Seed mahasiswa: Muhammad Raihan Zaldiputra")

    logger.info(f"✅ Database ready: {DB_NAME}@{DB_HOST}:{DB_PORT}")


# ────────────────────────────────────────────────
# Admin User Functions
# ────────────────────────────────────────────────
@_timed_db_op
def get_admin_by_username(username: str) -> Optional[Dict]:
    """Cari admin berdasarkan username. Returns dict atau None."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT id, username, email, password_hash, is_active FROM admin_users WHERE username = %s",
                (username,)
            )
            return cursor.fetchone()


# ────────────────────────────────────────────────────────
# Lookup Functions
# ────────────────────────────────────────────────────────
@_timed_db_op
def lookup_mahasiswa(nim: str) -> Optional[Dict]:
    """Cari mahasiswa berdasarkan NIM. Returns dict atau None."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM mahasiswa WHERE nim = %s", (nim,))
            return cursor.fetchone()


def fuzzy_name_match(ocr_name: str, db_name: str) -> float:
    """
    Hitung kesamaan nama dari OCR vs database.
    Returns float 0.0 - 1.0.
    """
    if not ocr_name or not db_name:
        return 0.0
    return SequenceMatcher(
        None,
        ocr_name.lower().strip(),
        db_name.lower().strip(),
    ).ratio()


@_timed_db_op
def verify_student(nim: str, ocr_name: Optional[str] = None) -> Dict:
    """
    Verifikasi mahasiswa — lookup NIM dan fuzzy match nama.

    Returns:
        {
            "success": bool,
            "verified": bool,
            "nim": str,
            "nama_db": str | None,
            "nama_ocr": str | None,
            "name_similarity": float,
            "prodi": str | None,
            "angkatan": int | None,
            "status": str | None,
            "message": str,
        }
    """
    result = {
        "success": False,
        "verified": False,
        "nim": nim,
        "nama_db": None,
        "nama_ocr": ocr_name,
        "name_similarity": 0.0,
        "prodi": None,
        "angkatan": None,
        "status": None,
        "message": "",
    }

    # Step 1: NIM lookup
    student = lookup_mahasiswa(nim)
    if not student:
        result["message"] = f"❌ NIM {nim} tidak terdaftar di sistem"
        logger.warning(result["message"])
        return result

    result["nama_db"] = student["nama"]
    result["prodi"] = student["prodi"]
    result["angkatan"] = student["angkatan"]
    result["status"] = student["status"]

    # Step 2: Status check
    if student["status"] != "aktif":
        result["message"] = f"⚠️ Mahasiswa {nim} status: {student['status']}"
        logger.warning(result["message"])
        return result

    # Step 3: Fuzzy name match [M1] — pakai named constant
    if ocr_name:
        similarity = fuzzy_name_match(ocr_name, student["nama"])
        result["name_similarity"] = round(similarity, 2)

        if similarity < NAME_MATCH_THRESHOLD:
            result["message"] = (
                f"⚠️ Nama tidak cocok: OCR='{ocr_name}' vs DB='{student['nama']}' "
                f"(similarity: {similarity:.0%})"
            )
            logger.warning(result["message"])
            return result

    # Verified!
    result["success"] = True
    result["verified"] = True
    result["message"] = f"✅ Verified: {student['nama']} ({nim})"
    logger.info(result["message"])
    return result


# ────────────────────────────────────────────────────────
# Peminjaman (Lab Borrowing) Functions
# ────────────────────────────────────────────────────────
def _format_duration(start_str: str, end_str: str) -> str:
    """
    Hitung durasi antara dua timestamp string.
    [R2] Menggunakan total_seconds() — benar untuk durasi >24 jam.
    """
    masuk = datetime.strptime(start_str, TIMESTAMP_FORMAT)
    keluar = datetime.strptime(end_str, TIMESTAMP_FORMAT)
    total_secs = int((keluar - masuk).total_seconds())
    hours = total_secs // 3600
    minutes = (total_secs % 3600) // 60
    return f"{hours}j {minutes}m"


@_timed_db_op
def has_active_peminjaman(nim: str) -> bool:
    """Cek apakah mahasiswa sedang memiliki peminjaman aktif atau menunggu."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT id FROM peminjaman WHERE nim = %s AND status IN ('aktif', 'menunggu')",
                (nim,)
            )
            return cursor.fetchone() is not None


@_timed_db_op
def check_in(nim: str, lab: str = DEFAULT_LAB) -> Dict:
    """
    Catat mahasiswa masuk lab.
    Sesi awal berstatus 'menunggu' (harus di-ACC admin).
    """
    with get_connection() as conn:
        with conn.cursor() as cursor:
            # Cek peminjaman aktif/menunggu
            cursor.execute(
                "SELECT id, status, waktu_masuk FROM peminjaman WHERE nim = %s AND status IN ('aktif', 'menunggu')",
                (nim,)
            )
            active = cursor.fetchone()

            if active:
                state_msg = "di dalam lab" if active['status'] == 'aktif' else "menunggu ACC admin"
                return {
                    "success": False,
                    "message": f"⚠️ {nim} sudah {state_msg} sejak {active['waktu_masuk']}",
                    "peminjaman_id": active["id"],
                }

            # Insert peminjaman baru (status 'menunggu')
            now = datetime.now().strftime(TIMESTAMP_FORMAT)
            cursor.execute(
                "INSERT INTO peminjaman (nim, lab, waktu_masuk, status) VALUES (%s, %s, %s, 'menunggu')",
                (nim, lab, now)
            )
            pid = cursor.lastrowid

    logger.info(f"⏳ Check-in pending ACC: {nim} → {lab} (ID: {pid})")
    return {
        "success": True,
        "message": f"⏳ Verifikasi Wajah Berhasil! Menunggu persetujuan Admin lab.",
        "peminjaman_id": pid,
        "waktu_masuk": now,
    }

@_timed_db_op
def approve_peminjaman(pid: int) -> Dict:
    """Ubah status peminjaman dari 'menunggu' menjadi 'aktif'."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("UPDATE peminjaman SET status = 'aktif' WHERE id = %s AND status = 'menunggu'", (pid,))
            if cursor.rowcount == 0:
                 return {"success": False, "message": "Peminjaman tidak ditemukan atau sudah aktif/selesai."}
    
    logger.info(f"✅ Approve peminjaman ID: {pid}")
    return {"success": True, "message": "Peminjaman disetujui (Aktif)."}

@_timed_db_op
def reject_peminjaman(pid: int) -> Dict:
    """Tolak peminjaman, ubah status dari 'menunggu' menjadi 'ditolak'."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            now = datetime.now().strftime(TIMESTAMP_FORMAT)
            cursor.execute(
                "UPDATE peminjaman SET status = 'ditolak', waktu_keluar = %s WHERE id = %s AND status = 'menunggu'", 
                (now, pid)
            )
            if cursor.rowcount == 0:
                 return {"success": False, "message": "Peminjaman tidak ditemukan atau sudah aktif/selesai."}
    
    logger.warning(f"❌ Reject peminjaman ID: {pid}")
    return {"success": True, "message": "Peminjaman ditolak."}


@_timed_db_op
def check_out(nim: str) -> Dict:
    """
    Catat mahasiswa keluar lab.
    [R2] Duration fix: total_seconds().
    """
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT id, lab, waktu_masuk FROM peminjaman WHERE nim = %s AND status = 'aktif'",
                (nim,)
            )
            active = cursor.fetchone()

            if not active:
                return {
                    "success": False,
                    "message": f"⚠️ {nim} tidak memiliki peminjaman aktif",
                }

            now = datetime.now().strftime(TIMESTAMP_FORMAT)

            # Handle waktu_masuk yang bisa datetime object atau string
            waktu_masuk = active["waktu_masuk"]
            if isinstance(waktu_masuk, datetime):
                waktu_masuk_str = waktu_masuk.strftime(TIMESTAMP_FORMAT)
            else:
                waktu_masuk_str = str(waktu_masuk)

            cursor.execute(
                "UPDATE peminjaman SET waktu_keluar = %s, status = 'selesai' WHERE id = %s",
                (now, active["id"])
            )

    durasi = _format_duration(waktu_masuk_str, now)
    logger.info(f"📤 Check-out: {nim} (durasi: {durasi})")
    return {
        "success": True,
        "message": f"📤 Check-out! Durasi: {durasi}",
        "peminjaman_id": active["id"],
        "durasi": durasi,
    }


@_timed_db_op
def get_active_peminjaman() -> List[Dict]:
    """List semua peminjaman yang aktif atau sedang menunggu ACC."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT p.*, m.nama
                FROM peminjaman p
                JOIN mahasiswa m ON p.nim = m.nim
                WHERE p.status IN ('aktif', 'menunggu')
                ORDER BY p.waktu_masuk DESC
            """)
            return cursor.fetchall()


@_timed_db_op
def reset_all_peminjaman() -> Dict:
    """Reset semua peminjaman aktif → selesai. Untuk testing."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE peminjaman SET status = 'selesai', waktu_keluar = %s WHERE status = 'aktif'",
                (datetime.now().strftime(TIMESTAMP_FORMAT),)
            )
            count = cursor.rowcount

    logger.info(f"🔄 Reset: {count} peminjaman aktif → selesai")
    return {"success": True, "message": f"🔄 Reset {count} peminjaman aktif", "count": count}


# ────────────────────────────────────────────────────────
# Face Encoding Storage
# ────────────────────────────────────────────────────────
@_timed_db_op
def save_face_encoding(nim: str, encoding: np.ndarray) -> Dict:
    """
    Simpan face encoding (512-D numpy array) ke database.
    Encoding disimpan sebagai LONGBLOB (raw bytes, ~2KB).
    TIDAK menyimpan foto — hanya angka encoding.
    """
    with get_connection() as conn:
        encoding_bytes = encoding.astype(np.float32).tobytes()
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE mahasiswa SET face_encoding = %s WHERE nim = %s",
                (encoding_bytes, nim)
            )

    logger.info(f"✅ Face encoding saved for {nim} ({len(encoding_bytes)} bytes)")
    return {"success": True, "message": f"✅ Wajah terdaftar untuk {nim}"}


@_timed_db_op
def load_face_encoding(nim: str) -> Optional[np.ndarray]:
    """
    Load face encoding dari database.
    Returns 512-D numpy array (float32) atau None jika belum ada.
    """
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT face_encoding FROM mahasiswa WHERE nim = %s", (nim,)
            )
            row = cursor.fetchone()

    if row and row["face_encoding"]:
        encoding = np.frombuffer(row["face_encoding"], dtype=np.float32)
        logger.debug(f"  Face encoding loaded for {nim} ({len(encoding)}D)")
        return encoding

    return None


@_timed_db_op
def has_face_encoding(nim: str) -> bool:
    """Check apakah mahasiswa sudah punya face encoding."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT face_encoding FROM mahasiswa WHERE nim = %s", (nim,)
            )
            row = cursor.fetchone()
    return row is not None and row["face_encoding"] is not None


@_timed_db_op
def delete_face_encoding(nim: str) -> Dict[str, Any]:
    """Menghapus face encoding mahasiswa dari database (reset)."""
    if not verify_student(nim)["verified"]:
        return {"success": False, "message": "NIM tidak ditemukan."}

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("UPDATE mahasiswa SET face_encoding = NULL WHERE nim = %s", (nim,))

    logger.info(f"🗑 Face encoding dihapus untuk {nim}")
    return {"success": True, "message": f"Wajah untuk {nim} berhasil direset."}
