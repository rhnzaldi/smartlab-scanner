"""
Smart-Lab SV IPB — Seed Data Mahasiswa
Jalankan script ini untuk menambahkan data admin dan mahasiswa ke database MySQL.

Usage:
    python seed_mahasiswa.py --file students.xlsx
    python seed_mahasiswa.py --file students.csv

Format Excel/CSV:
    no | nim | nama | prodi | angkatan

Contoh:
    1 J0403231001 Alex Teknologi Rekayasa Perangkat Lunak 2023

Kolom `angkatan` tidak wajib; default 2023.
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

from db.database import init_db, get_connection
from security import get_password_hash

DEFAULT_ANGKATAN = 2023
DEFAULT_FILE = "students.xlsx"

# ════════════════════════════════════════════════════════
# Seed data default admin
# ════════════════════════════════════════════════════════
ADMIN_USERS = [
    {
        "username": "admin",
        "email": "admin@smartlab.id",
        "password": "admin123",
        "role": "admin",
    }
]


def normalize_student_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    normalized = {str(k).strip().lower(): v for k, v in row.items() if k is not None}
    nim = str(normalized.get("nim", "")).strip()
    if not nim:
        return None

    nama = str(normalized.get("nama", "")).strip() or f"Mahasiswa {nim.upper()}"
    prodi = str(normalized.get("prodi", "")).strip() or "TPL"
    angkatan_raw = normalized.get("angkatan") or normalized.get("tahun")
    try:
        angkatan = int(str(angkatan_raw).strip()) if angkatan_raw not in (None, "") else DEFAULT_ANGKATAN
    except Exception:
        angkatan = DEFAULT_ANGKATAN

    return {
        "nim": nim.upper(),
        "nama": nama,
        "prodi": prodi,
        "angkatan": angkatan,
    }


def load_students_from_csv(path: str) -> List[Dict[str, Any]]:
    students: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            student = normalize_student_row(row)
            if student:
                students.append(student)
    return students


def load_students_from_excel(path: str) -> List[Dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Library pandas belum terpasang. Install dengan: pip install pandas openpyxl"
        ) from exc

    df = pd.read_excel(path, dtype=str)
    df = df.fillna("")
    students: List[Dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        student = normalize_student_row(row)
        if student:
            students.append(student)
    return students


def load_students_from_file(path: str) -> List[Dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return load_students_from_csv(path)
    if ext in (".xlsx", ".xls"):
        return load_students_from_excel(path)
    raise ValueError("File harus .xlsx, .xls, atau .csv")


def seed(students_file: str) -> None:
    init_db()
    students = load_students_from_file(students_file)

    if not students:
        print(f"Tidak ditemukan data mahasiswa di {students_file}")
        return

    with get_connection() as conn:
        with conn.cursor() as cursor:
            inserted = 0
            skipped = 0

            # Seed admin users
            for admin in ADMIN_USERS:
                cursor.execute(
                    "SELECT username FROM admin_users WHERE username = %s",
                    (admin["username"],)
                )
                existing_admin = cursor.fetchone()
                if existing_admin:
                    print(f"  ⏭️  admin {admin['username']} (sudah ada)")
                else:
                    cursor.execute(
                        "INSERT INTO admin_users (username, email, password_hash, role) VALUES (%s, %s, %s, %s)",
                        (
                            admin["username"],
                            admin["email"],
                            get_password_hash(admin["password"]),
                            admin["role"],
                        ),
                    )
                    print(f"  ✅ admin {admin['username']}")

            # Seed mahasiswa
            for student in students:
                nim = student["nim"]
                nama = student["nama"]
                prodi = student["prodi"]
                angkatan = student["angkatan"]

                cursor.execute(
                    "SELECT nim FROM mahasiswa WHERE UPPER(nim) = UPPER(%s)",
                    (nim,)
                )
                existing = cursor.fetchone()

                if existing:
                    print(f"  ⏭️  {nim} — {nama} (sudah ada)")
                    skipped += 1
                else:
                    cursor.execute(
                        "INSERT INTO mahasiswa (nim, nama, prodi, angkatan, status, password_hash) VALUES (%s, %s, %s, %s, 'aktif', %s)",
                        (nim, nama, prodi, angkatan, get_password_hash(nim.lower())),
                    )
                    print(f"  ✅ {nim} — {nama}")
                    inserted += 1

    print(f"\nSelesai! {inserted} ditambahkan, {skipped} sudah ada.")

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT nim, nama, prodi, angkatan FROM mahasiswa ORDER BY nim")
            rows = cursor.fetchall()
            print(f"\n📋 Total mahasiswa: {len(rows)}")
            print(f"{'NIM':<15} {'Nama':<35} {'Prodi':<20} {'Angkatan'}")
            print("─" * 90)
            for r in rows:
                print(f"{r['nim']:<15} {r['nama']:<35} {r['prodi']:<20} {r['angkatan']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed admin and mahasiswa data from Excel/CSV.")
    parser.add_argument(
        "--file",
        default=DEFAULT_FILE,
        help="Path ke file students.xlsx atau students.csv",
    )
    args = parser.parse_args()
    seed(args.file)
