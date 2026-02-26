"""
Smart-Lab SV IPB — Database Module
SQLite database untuk manajemen mahasiswa dan peminjaman lab.

NFR Refactored:
- [R1] Context manager untuk semua koneksi (no connection leak)
- [R2] Duration fix: total_seconds() bukan .seconds
- [M1] Named constants (no magic numbers)
- [M7] Consistent return types
- [O2] DB operation timing via decorator
"""

import sqlite3
import logging
import os
import functools
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, Dict, List
from difflib import SequenceMatcher

import numpy as np

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────
# Constants (menggantikan magic numbers) [M1]
# ────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "smartlab.db")
NAME_MATCH_THRESHOLD = 0.5       # minimum similarity untuk fuzzy name match
NIM_PATTERN = r"^J[A-Z0-9]{6,12}$"  # valid NIM format untuk input validation [S2]
DEFAULT_LAB = "Lab Smart-Lab"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


# ────────────────────────────────────────────────────────
# Connection Management [R1]
# ────────────────────────────────────────────────────────
@contextmanager
def get_connection():
    """
    Context manager untuk SQLite connection.
    Otomatis commit saat sukses, rollback saat error, selalu close.

    Usage:
        with get_connection() as conn:
            conn.execute("SELECT ...")
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")  # enforce FK constraints
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


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
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mahasiswa (
                nim TEXT PRIMARY KEY,
                nama TEXT NOT NULL,
                prodi TEXT,
                angkatan INTEGER,
                status TEXT DEFAULT 'aktif',
                face_encoding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Migration: add face_encoding column if table already exists without it
        try:
            cursor.execute("ALTER TABLE mahasiswa ADD COLUMN face_encoding BLOB")
            logger.info("✅ Added face_encoding column to mahasiswa")
        except sqlite3.OperationalError:
            pass  # column already exists

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS peminjaman (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nim TEXT NOT NULL REFERENCES mahasiswa(nim),
                lab TEXT NOT NULL DEFAULT 'Lab Smart-Lab',
                waktu_masuk TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                waktu_keluar TIMESTAMP,
                status TEXT DEFAULT 'aktif',
                scan_confidence REAL,
                catatan TEXT
            )
        """)

        # Index untuk query yang sering dipakai [P1]
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_peminjaman_nim_status
            ON peminjaman(nim, status)
        """)

        # Seed data
        cursor.execute("SELECT COUNT(*) FROM mahasiswa")
        if cursor.fetchone()[0] == 0:
            cursor.execute("""
                INSERT INTO mahasiswa (nim, nama, prodi, angkatan, status)
                VALUES (?, ?, ?, ?, ?)
            """, (
                "J0403231061",
                "Muhammad Raihan Zaldiputra",
                "TPL",
                2023,
                "aktif",
            ))
            logger.info("✅ Seed data: Muhammad Raihan Zaldiputra (J0403231061)")

    logger.info(f"✅ Database ready: {DB_PATH}")


# ────────────────────────────────────────────────────────
# Lookup Functions
# ────────────────────────────────────────────────────────
@_timed_db_op
def lookup_mahasiswa(nim: str) -> Optional[Dict]:
    """Cari mahasiswa berdasarkan NIM. Returns dict atau None."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM mahasiswa WHERE nim = ?", (nim,))
        row = cursor.fetchone()
        return dict(row) if row else None


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
            "success": bool,      # [M7] konsisten dengan fungsi lain
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
def check_in(nim: str, lab: str = DEFAULT_LAB) -> Dict:
    """
    Catat mahasiswa masuk lab.
    [R1] Context manager untuk auto-close connection.
    """
    with get_connection() as conn:
        cursor = conn.cursor()

        # Cek peminjaman aktif
        cursor.execute(
            "SELECT id, waktu_masuk FROM peminjaman WHERE nim = ? AND status = 'aktif'",
            (nim,)
        )
        active = cursor.fetchone()

        if active:
            return {
                "success": False,
                "message": f"⚠️ {nim} sudah check-in sejak {active['waktu_masuk']}",
                "peminjaman_id": active["id"],
            }

        # Insert peminjaman baru
        now = datetime.now().strftime(TIMESTAMP_FORMAT)
        cursor.execute(
            "INSERT INTO peminjaman (nim, lab, waktu_masuk, status) VALUES (?, ?, ?, 'aktif')",
            (nim, lab, now)
        )
        pid = cursor.lastrowid

    logger.info(f"📥 Check-in: {nim} → {lab} (ID: {pid})")
    return {
        "success": True,
        "message": f"📥 Check-in berhasil! Selamat datang di {lab}",
        "peminjaman_id": pid,
        "waktu_masuk": now,
    }


@_timed_db_op
def check_out(nim: str) -> Dict:
    """
    Catat mahasiswa keluar lab.
    [R2] Duration fix: total_seconds().
    """
    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id, lab, waktu_masuk FROM peminjaman WHERE nim = ? AND status = 'aktif'",
            (nim,)
        )
        active = cursor.fetchone()

        if not active:
            return {
                "success": False,
                "message": f"⚠️ {nim} tidak memiliki peminjaman aktif",
            }

        now = datetime.now().strftime(TIMESTAMP_FORMAT)
        cursor.execute(
            "UPDATE peminjaman SET waktu_keluar = ?, status = 'selesai' WHERE id = ?",
            (now, active["id"])
        )

    durasi = _format_duration(active["waktu_masuk"], now)
    logger.info(f"📤 Check-out: {nim} (durasi: {durasi})")
    return {
        "success": True,
        "message": f"📤 Check-out! Durasi: {durasi}",
        "peminjaman_id": active["id"],
        "durasi": durasi,
    }


@_timed_db_op
def get_active_peminjaman() -> List[Dict]:
    """List semua peminjaman aktif."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT p.*, m.nama
            FROM peminjaman p
            JOIN mahasiswa m ON p.nim = m.nim
            WHERE p.status = 'aktif'
            ORDER BY p.waktu_masuk DESC
        """)
        return [dict(r) for r in cursor.fetchall()]


@_timed_db_op
def reset_all_peminjaman() -> Dict:
    """Reset semua peminjaman aktif → selesai. Untuk testing."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE peminjaman SET status = 'selesai', waktu_keluar = ? WHERE status = 'aktif'",
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
    Simpan face encoding (128-D numpy array) ke database.
    Encoding disimpan sebagai BLOB (raw bytes, 512 bytes).
    TIDAK menyimpan foto — hanya angka encoding.
    """
    with get_connection() as conn:
        encoding_bytes = encoding.astype(np.float64).tobytes()
        conn.execute(
            "UPDATE mahasiswa SET face_encoding = ? WHERE nim = ?",
            (encoding_bytes, nim)
        )

    logger.info(f"✅ Face encoding saved for {nim} ({len(encoding_bytes)} bytes)")
    return {"success": True, "message": f"✅ Wajah terdaftar untuk {nim}"}


@_timed_db_op
def load_face_encoding(nim: str) -> Optional[np.ndarray]:
    """
    Load face encoding dari database.
    Returns 128-D numpy array atau None jika belum ada.
    """
    with get_connection() as conn:
        row = conn.execute(
            "SELECT face_encoding FROM mahasiswa WHERE nim = ?", (nim,)
        ).fetchone()

    if row and row["face_encoding"]:
        encoding = np.frombuffer(row["face_encoding"], dtype=np.float64)
        logger.debug(f"  Face encoding loaded for {nim} ({len(encoding)}D)")
        return encoding

    return None


@_timed_db_op
def has_face_encoding(nim: str) -> bool:
    """Check apakah mahasiswa sudah punya face encoding."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT face_encoding FROM mahasiswa WHERE nim = ?", (nim,)
        ).fetchone()
    return row is not None and row["face_encoding"] is not None
