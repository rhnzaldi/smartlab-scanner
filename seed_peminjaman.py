#!/usr/bin/env python3
"""
Seed contoh peminjaman selaras dengan lab & jadwal di database.

NIM yang dipakai (urutan baris = urutan slot jadwal):
  J0403231105, J0403231001, J0403231061

Syarat:
  - Tabel `jadwal` punya minimal 1 baris aktif (idealnya 3 untuk tiga contoh).
  - Mahasiswa dengan NIM di atas akan dibuat otomatis jika belum ada (FK).

Usage:
    python seed_peminjaman.py
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, time
from typing import Any, List, Tuple

from dotenv import load_dotenv

load_dotenv()

from db.database import get_connection, init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("seed_peminjaman")

# Urutan: baris peminjaman ke-1, ke-2, ke-3 mengikuti slot jadwal ke-1, ke-2, ke-3
SEED_NIMS: Tuple[str, ...] = ("J0403231105", "J0403231001", "J0403231061")

# Python: Senin = 0 … Minggu = 6 — sama dengan seed_db (hari akademik)
HARI_TO_WEEKDAY = {"Senin": 0, "Selasa": 1, "Rabu": 2, "Kamis": 3, "Jumat": 4, "Sabtu": 5}


def _parse_time(val: Any) -> Tuple[int, int]:
    """Jam mulai/selesai dari MySQL bisa TIME → timedelta atau datetime.time."""
    if val is None:
        return 9, 0
    if isinstance(val, timedelta):
        total = int(val.total_seconds()) % 86400
        return total // 3600, (total % 3600) // 60
    if hasattr(val, "hour") and hasattr(val, "minute"):
        return int(val.hour), int(val.minute)
    s = str(val)
    parts = s.replace(".", ":").split(":")
    h = int(parts[0]) if parts else 9
    m = int(parts[1]) if len(parts) > 1 else 0
    return h, m


def next_datetime_in_slot(hari: str, hour: int, minute: int) -> datetime:
    """Ambil datetime pada hari `hari` jam (hour, minute); jika sudah lewat minggu ini → minggu depan."""
    wd = HARI_TO_WEEKDAY.get(str(hari).strip())
    if wd is None:
        wd = 0
    now = datetime.now()
    delta_days = (wd - now.weekday()) % 7
    target_date = (now + timedelta(days=delta_days)).date()
    dt = datetime.combine(target_date, time(hour=hour, minute=minute))
    if dt <= now:
        dt += timedelta(days=7)
    return dt


def ensure_seed_mahasiswa(cursor, nims: List[str]) -> None:
    """Pastikan setiap NIM seed ada di tabel mahasiswa (untuk FK peminjaman)."""
    for nim in nims:
        cursor.execute("SELECT nim FROM mahasiswa WHERE nim = %s", (nim,))
        if cursor.fetchone():
            continue
        cursor.execute(
            """
            INSERT INTO mahasiswa (nim, nama, prodi, angkatan, status)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                nim,
                f"Mahasiswa seed {nim}",
                "Teknologi Rekayasa Perangkat Lunak",
                2023,
                "aktif",
            ),
        )
        logger.info("Menambah mahasiswa %s untuk seed peminjaman.", nim)


def seed_peminjaman() -> None:
    init_db()

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS cnt FROM peminjaman")
            if cursor.fetchone()["cnt"] > 0:
                logger.info("Tabel peminjaman sudah berisi — seed dilewati.")
                return

            cursor.execute(
                """
                SELECT id, lab, gedung, hari, jam_mulai, jam_selesai
                FROM jadwal
                WHERE is_archived = FALSE
                ORDER BY id
                LIMIT 3
                """
            )
            slots = cursor.fetchall()

            if len(slots) < 1:
                logger.error(
                    "Butuh minimal 1 entri jadwal aktif. Jalankan `python seed_db.py` terlebih dahulu."
                )
                return

            count = min(len(slots), len(SEED_NIMS))
            nims_to_use = list(SEED_NIMS[:count])
            ensure_seed_mahasiswa(cursor, nims_to_use)

            inserts = []
            for idx in range(count):
                slot = slots[idx]
                nim = nims_to_use[idx]
                lab = slot["lab"]
                hari = slot["hari"]
                sh, sm = _parse_time(slot["jam_mulai"])
                eh, em = _parse_time(slot["jam_selesai"])
                start_min = sh * 60 + sm
                end_min = eh * 60 + em
                mid_min = (start_min + end_min) // 2
                hh, mm = mid_min // 60, mid_min % 60
                if mid_min <= start_min:
                    hh, mm = sh, sm

                waktu_masuk = next_datetime_in_slot(hari, hh, mm)
                # Baris pertama aktif, sisanya menunggu (cocok untuk uji dashboard + pending)
                status = "aktif" if idx == 0 else "menunggu"
                catatan = f"Seed: jadwal id={slot['id']} ({hari}, {lab}) — NIM {nim}"
                inserts.append((nim, lab, waktu_masuk, status, catatan))

            for nim, lab, waktu_masuk, status, catatan in inserts:
                cursor.execute(
                    """
                    INSERT INTO peminjaman (nim, lab, waktu_masuk, status, catatan)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (nim, lab, waktu_masuk, status, catatan),
                )

            logger.info(
                "Berhasil menyisipkan %s peminjaman: %s",
                len(inserts),
                [(i[0], i[1], i[2], i[3]) for i in inserts],
            )


if __name__ == "__main__":
    seed_peminjaman()
