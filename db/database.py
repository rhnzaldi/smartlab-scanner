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
import json
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

            # Tabel labs (untuk manajemen laboratorium)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS labs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    location VARCHAR(100) NOT NULL,
                    capacity INT DEFAULT 0,
                    op_start TIME,
                    op_end TIME,
                    use_start TIME,
                    use_end TIME,
                    equipment JSON,
                    status_override VARCHAR(20),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)

            # Tabel jadwal (schedule)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS jadwal (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    mata_kuliah VARCHAR(150) NOT NULL,
                    kelas VARCHAR(50) NOT NULL,
                    prodi VARCHAR(100) NOT NULL,
                    lab VARCHAR(100) NOT NULL,
                    gedung VARCHAR(100) NOT NULL,
                    hari VARCHAR(20) NOT NULL,
                    jam_mulai TIME NOT NULL,
                    jam_selesai TIME NOT NULL,
                    tipe_semester VARCHAR(50) NOT NULL,
                    tahun_ajaran VARCHAR(20) NOT NULL,
                    status VARCHAR(20) DEFAULT 'tersedia',
                    is_archived BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)


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
# Lab Management Functions
# ────────────────────────────────────────────────────────
@_timed_db_op
def get_labs() -> List[Dict]:
    """Ambil daftar laboratorium."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM labs ORDER BY id")
            labs = cursor.fetchall()

    # Convert JSON string to list if needed, and format timedeltas
    for lab in labs:
        if lab.get("equipment") and isinstance(lab["equipment"], str):
            try:
                lab["equipment"] = json.loads(lab["equipment"])
            except Exception:
                lab["equipment"] = []
                
        # Format timedelta properly to HH:MM
        for key in ["op_start", "op_end", "use_start", "use_end"]:
            if lab.get(key) is not None:
                s = str(lab[key])
                if len(s) == 7: s = "0" + s
                lab[key] = s[:5] if len(s) >= 5 else s
    
    return labs


@_timed_db_op
def get_lab(lab_id: int) -> Optional[Dict]:
    """Ambil detail satu laboratorium."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM labs WHERE id = %s", (lab_id,))
            lab = cursor.fetchone()

    if not lab:
        return None

    if lab.get("equipment") and isinstance(lab["equipment"], str):
        try:
            lab["equipment"] = json.loads(lab["equipment"])
        except Exception:
            lab["equipment"] = []
            
    # Format timedelta properly to HH:MM
    for key in ["op_start", "op_end", "use_start", "use_end"]:
        if lab.get(key) is not None:
            s = str(lab[key])
            if len(s) == 7: s = "0" + s
            lab[key] = s[:5] if len(s) >= 5 else s

    return lab


@_timed_db_op
def create_lab(
    name: str,
    location: str,
    capacity: int,
    op_start: str,
    op_end: str,
    use_start: str,
    use_end: str,
    equipment: List[str],
    status_override: Optional[str] = None,
) -> Dict:
    """Buat lab baru."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                    INSERT INTO labs (name, location, capacity, op_start, op_end, use_start, use_end, equipment, status_override)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    name,
                    location,
                    capacity,
                    op_start,
                    op_end,
                    use_start,
                    use_end,
                    json.dumps(equipment),
                    status_override,
                ),
            )
            lab_id = cursor.lastrowid

    return get_lab(lab_id)


@_timed_db_op
def update_lab(
    lab_id: int,
    name: str,
    location: str,
    capacity: int,
    op_start: str,
    op_end: str,
    use_start: str,
    use_end: str,
    equipment: List[str],
    status_override: Optional[str] = None,
) -> Optional[Dict]:
    """Perbarui data lab."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                    UPDATE labs
                    SET name=%s, location=%s, capacity=%s, op_start=%s, op_end=%s,
                        use_start=%s, use_end=%s, equipment=%s, status_override=%s
                    WHERE id=%s
                """,
                (
                    name,
                    location,
                    capacity,
                    op_start,
                    op_end,
                    use_start,
                    use_end,
                    json.dumps(equipment),
                    status_override,
                    lab_id,
                ),
            )
            # Not checking rowcount == 0 to return None, 
            # because MySQL rowcount is 0 if data isn't changed.
            
    return get_lab(lab_id)


@_timed_db_op
def delete_lab(lab_id: int) -> bool:
    """Hapus lab."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM labs WHERE id = %s", (lab_id,))
            return cursor.rowcount > 0


# ────────────────────────────────────────────────────────
# Schedule (Jadwal) Functions
# ────────────────────────────────────────────────────────
@_timed_db_op
def get_jadwal(include_archived: bool = False) -> List[Dict]:
    """Ambil jadwal laboratorium.

    Args:
        include_archived: Jika True, kembalikan semua jadwal (termasuk yang diarsipkan).
    """
    with get_connection() as conn:
        with conn.cursor() as cursor:
            if include_archived:
                cursor.execute("SELECT * FROM jadwal ORDER BY id")
            else:
                cursor.execute("SELECT * FROM jadwal WHERE is_archived = FALSE ORDER BY id")
            jadwal_list = cursor.fetchall()

    for jadwal in jadwal_list:
        if jadwal.get("jam_mulai") is not None:
            s = str(jadwal["jam_mulai"])
            if len(s) == 7: s = "0" + s
            jadwal["jam_mulai"] = s[:5] if len(s) >= 5 else s
        if jadwal.get("jam_selesai") is not None:
            s = str(jadwal["jam_selesai"])
            if len(s) == 7: s = "0" + s
            jadwal["jam_selesai"] = s[:5] if len(s) >= 5 else s

    return jadwal_list


@_timed_db_op
def get_jadwal_item(jadwal_id: int) -> Optional[Dict]:
    """Ambil satu entri jadwal berdasarkan ID."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM jadwal WHERE id = %s", (jadwal_id,))
            jadwal = cursor.fetchone()

    if jadwal:
        if jadwal.get("jam_mulai") is not None:
            s = str(jadwal["jam_mulai"])
            if len(s) == 7: s = "0" + s
            jadwal["jam_mulai"] = s[:5] if len(s) >= 5 else s
        if jadwal.get("jam_selesai") is not None:
            s = str(jadwal["jam_selesai"])
            if len(s) == 7: s = "0" + s
            jadwal["jam_selesai"] = s[:5] if len(s) >= 5 else s

    return jadwal


@_timed_db_op
def create_jadwal(
    mata_kuliah: str,
    kelas: str,
    prodi: str,
    lab: str,
    gedung: str,
    hari: str,
    jam_mulai: str,
    jam_selesai: str,
    tipe_semester: str,
    tahun_ajaran: str,
    status: str = "tersedia",
) -> Dict:
    """Buat entri jadwal baru."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                    INSERT INTO jadwal (mata_kuliah, kelas, prodi, lab, gedung, hari, jam_mulai, jam_selesai, tipe_semester, tahun_ajaran, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    mata_kuliah,
                    kelas,
                    prodi,
                    lab,
                    gedung,
                    hari,
                    jam_mulai,
                    jam_selesai,
                    tipe_semester,
                    tahun_ajaran,
                    status,
                ),
            )
            jadwal_id = cursor.lastrowid

    return get_jadwal_item(jadwal_id)


@_timed_db_op
def update_jadwal(
    jadwal_id: int,
    mata_kuliah: str,
    kelas: str,
    prodi: str,
    lab: str,
    gedung: str,
    hari: str,
    jam_mulai: str,
    jam_selesai: str,
    tipe_semester: str,
    tahun_ajaran: str,
    status: str,
) -> Optional[Dict]:
    """Update entri jadwal."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                    UPDATE jadwal
                    SET mata_kuliah=%s, kelas=%s, prodi=%s, lab=%s, gedung=%s,
                        hari=%s, jam_mulai=%s, jam_selesai=%s, tipe_semester=%s, tahun_ajaran=%s, status=%s
                    WHERE id=%s
                """,
                (
                    mata_kuliah,
                    kelas,
                    prodi,
                    lab,
                    gedung,
                    hari,
                    jam_mulai,
                    jam_selesai,
                    tipe_semester,
                    tahun_ajaran,
                    status,
                    jadwal_id,
                ),
            )
            if cursor.rowcount == 0:
                return None

    return get_jadwal_item(jadwal_id)


@_timed_db_op
def delete_jadwal(jadwal_id: int) -> bool:
    """Hapus jadwal."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM jadwal WHERE id = %s", (jadwal_id,))
            return cursor.rowcount > 0


@_timed_db_op
def archive_all_jadwal() -> int:
    """Archive semua jadwal (reset semester)."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("UPDATE jadwal SET is_archived = TRUE WHERE is_archived = FALSE")
            return cursor.rowcount


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
            
            # Fetch lab info of the approved active session
            cursor.execute("SELECT lab FROM peminjaman WHERE id = %s", (pid,))
            lab_res = cursor.fetchone()
            if lab_res:
                lab = lab_res["lab"]
                # Coba temukan jadwal aktif saat ini untuk lab tersebut dan tandai digunakan
                now_str = datetime.now().strftime("%H:%M:%S")
                # Karena MariaDB time comparison bisa simple, kita pakai nama hari bahasa Indonesia (opsional, jika jadwal mengikat hari)
                # Mari kita update semua jadwal untuk lab tsb di waktu skr
                cursor.execute(
                    "UPDATE jadwal SET status = 'digunakan' WHERE lab = %s AND jam_mulai <= %s AND jam_selesai >= %s",
                    (lab, now_str, now_str)
                )
    
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
                 
            # Note: No need to update jadwal status to digunakan on reject, it stays tersedia/menunggu.
    
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
            
            # Revert lab schedule back to tersedia if there are no more active peminjaman for this lab
            cursor.execute("SELECT COUNT(id) AS c FROM peminjaman WHERE lab = %s AND status = 'aktif'", (active["lab"],))
            active_count = cursor.fetchone()
            if active_count and active_count["c"] == 0:
                now_time_str = datetime.now().strftime("%H:%M:%S")
                cursor.execute(
                    "UPDATE jadwal SET status = 'tersedia' WHERE lab = %s AND jam_mulai <= %s AND jam_selesai >= %s AND status = 'digunakan'",
                    (active["lab"], now_time_str, now_time_str)
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
# Reporting / History Functions
# ────────────────────────────────────────────────────────
@_timed_db_op
def get_peminjaman_history(
    year: Optional[int] = None,
    month: Optional[int] = None,
    prodi: Optional[str] = None,
) -> List[Dict]:
    """Ambil riwayat peminjaman (peminjaman selesai/ditolak) dengan filter opsional."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            query = """
                SELECT p.*, m.nama, m.prodi
                FROM peminjaman p
                LEFT JOIN mahasiswa m ON p.nim = m.nim
            """
            conditions = []
            params: List[Any] = []

            if year is not None:
                conditions.append("YEAR(p.waktu_masuk) = %s")
                params.append(year)
            if month is not None:
                conditions.append("MONTH(p.waktu_masuk) = %s")
                params.append(month)
            if prodi:
                conditions.append("m.prodi = %s")
                params.append(prodi)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY p.waktu_masuk DESC"
            cursor.execute(query, tuple(params))
            return cursor.fetchall()


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
