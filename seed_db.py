"""Smart-Lab SV IPB — Seed Data (Labs + Jadwal)

Jalankan script ini untuk menambahkan data awal ke tabel `labs` dan `jadwal`.

Usage:
    python seed_db.py
"""

import json
from dotenv import load_dotenv
load_dotenv()

from db.database import init_db, get_connection

# ════════════════════════════════════════════════════════
# DATA SEED (dari dump SQL)
# ════════════════════════════════════════════════════════
LABS = [
    {
        "name": "Lab Jaringan 01",
        "location": "Gedung Delta",
        "capacity": 25,
        "op_start": "07:00:00",
        "op_end": "18:00:00",
        "use_start": "08:00:00",
        "use_end": "11:00:00",
        "equipment": ["PC Intel Core i5", "Router Cisco"],
    },
    {"name": "CB Pemrograman", "location": "Gedung CB", "capacity": 36, "op_start": "07:00:00", "op_end": "18:00:00", "use_start": "08:00:00", "use_end": "11:00:00", "equipment": ["PC Core i7"]},
    {"name": "CA RPL", "location": "Gedung CA", "capacity": 36, "op_start": "07:00:00", "op_end": "18:00:00", "use_start": "08:00:00", "use_end": "11:00:00", "equipment": ["PC Core i7"]},
    {"name": "CB K-70", "location": "Gedung CB", "capacity": 24, "op_start": "07:00:00", "op_end": "18:00:00", "use_start": "08:00:00", "use_end": "11:00:00", "equipment": ["PC Core i7"]},
    {"name": "Lab Jaringan 02", "location": "Gedung Delta", "capacity": 24, "op_start": "07:00:00", "op_end": "18:00:00", "use_start": "08:00:00", "use_end": "11:00:00", "equipment": ["PC Core i7"]},
    {"name": "CB Data Science", "location": "Gedung CB", "capacity": 24, "op_start": "07:00:00", "op_end": "18:00:00", "use_start": "08:00:00", "use_end": "11:00:00", "equipment": ["PC Core i7"]},
    {"name": "CA KOM 1", "location": "Gedung CA", "capacity": 36, "op_start": "07:00:00", "op_end": "18:00:00", "use_start": "08:00:00", "use_end": "11:00:00", "equipment": ["PC Core i7"]},
    {"name": "CA KOM 2", "location": "Gedung CA", "capacity": 36, "op_start": "07:00:00", "op_end": "18:00:00", "use_start": "08:00:00", "use_end": "11:00:00", "equipment": ["PC Core i7"]},
]

JADWAL = [
    [
        "Visual Komputer Cerdas",
        "TPL-A",
        "Teknologi Rekayasa Perangkat Lunak",
        "CB K-70",
        "Gedung CA",
        "Kamis",
        "07:00:00",
        "11:00:00",
        "Genap",
        "2025/2026",
        "tersedia",
    ],
    [
        "Kecerdasan Bisnis",
        "TPL-A",
        "Teknologi Rekayasa Perangkat Lunak",
        "CA KOM 1",
        "Gedung CA",
        "Senin",
        "08:00:00",
        "12:00:00",
        "Genap",
        "2025/2026",
        "tersedia",
    ],
    [
        "Proyek Pengembangan Perangkat Lunak",
        "TPL-A",
        "Teknologi Rekayasa Perangkat Lunak",
        "CB Pemrograman",
        "Gedung CB",
        "Selasa",
        "07:00:00",
        "13:00:00",
        "Genap",
        "2025/2026",
        "tersedia",
    ],
    [
        "Teknologi Big Data",
        "TPL-A",
        "Teknologi Rekayasa Perangkat Lunak",
        "CA RPL",
        "Gedung CA",
        "Kamis",
        "16:00:00",
        "20:00:00",
        "Genap",
        "2025/2026",
        "tersedia",
    ],
    [
        "Teknologi Big Data",
        "TPL-B",
        "Teknologi Rekayasa Perangkat Lunak",
        "CA RPL",
        "Gedung CA",
        "Rabu",
        "07:00:00",
        "11:00:00",
        "Genap",
        "2025/2026",
        "tersedia",
    ],
    [
        "Visual Komputer Cerdas",
        "TPL-B",
        "Teknologi Rekayasa Perangkat Lunak",
        "CA RPL",
        "Gedung CA",
        "Kamis",
        "11:00:00",
        "15:00:00",
        "Genap",
        "2025/2026",
        "tersedia",
    ],
    [
        "Kecerdasan Bisnis",
        "TPL-B",
        "Teknologi Rekayasa Perangkat Lunak",
        "CA RPL",
        "Gedung CA",
        "Selasa",
        "15:00:00",
        "19:00:00",
        "Genap",
        "2025/2026",
        "tersedia",
    ],
    [
        "Proyek Pengembangan Perangkat Lunak",
        "TPL-B",
        "Teknologi Rekayasa Perangkat Lunak",
        "CA KOM 1",
        "Gedung CA",
        "Senin",
        "13:00:00",
        "19:00:00",
        "Genap",
        "2025/2026",
        "tersedia",
    ],
]


def seed_labs() -> None:
    """Seed tabel labs jika masih kosong."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS cnt FROM labs")
            if cursor.fetchone()["cnt"] > 0:
                print("✅ labs sudah terisi (skip seeding).")
                return

            inserted = 0
            for lab in LABS:
                cursor.execute(
                    "INSERT INTO labs (name, location, capacity, op_start, op_end, use_start, use_end, equipment, status_override) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        lab["name"],
                        lab["location"],
                        lab["capacity"],
                        lab["op_start"],
                        lab["op_end"],
                        lab["use_start"],
                        lab["use_end"],
                        json.dumps(lab["equipment"]),
                        None,
                    ),
                )
                inserted += 1

    print(f"✅ Seeded labs: {inserted} rows")


def seed_jadwal() -> None:
    """Seed tabel jadwal jika masih kosong."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS cnt FROM jadwal")
            if cursor.fetchone()["cnt"] > 0:
                print("✅ jadwal sudah terisi (skip seeding).")
                return

            inserted = 0
            for row in JADWAL:
                cursor.execute(
                    """
                        INSERT INTO jadwal (mata_kuliah, kelas, prodi, lab, gedung, hari, jam_mulai, jam_selesai, tipe_semester, tahun_ajaran, status)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    tuple(row),
                )
                inserted += 1

    print(f"✅ Seeded jadwal: {inserted} rows")


def seed_all() -> None:
    """Init DB (create tables) + seed labs + jadwal."""
    init_db()
    seed_labs()
    seed_jadwal()


if __name__ == "__main__":
    seed_all()
