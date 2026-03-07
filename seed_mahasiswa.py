"""
Smart-Lab SV IPB — Seed Data Mahasiswa
Jalankan script ini untuk menambahkan data mahasiswa ke database MySQL.

Usage:
    python seed_mahasiswa.py

Cara menambah mahasiswa:
    Tambahkan tuple baru di list MAHASISWA di bawah ini.
    Format: (NIM, Nama Lengkap, Prodi, Angkatan)
"""

from dotenv import load_dotenv
load_dotenv()

from db.database import init_db, get_connection

# ════════════════════════════════════════════════════════
# DAFTAR MAHASISWA — Tambahkan di sini
# Format: (NIM, Nama, Prodi, Angkatan)
# ════════════════════════════════════════════════════════
MAHASISWA = [
    ("J0403231061", "Muhammad Raihan Zaldiputra", "TPL", 2023),
    ("J0403231105", "Raffa Danendra Pramono", "TPL", 2023),
    ("J0403231052", "Keisyah Zahra Anatasya", "TPL", 2023),
    ("J0403231150", "Ahmad", "TPL", 2023),
    
]


def seed():
    init_db()

    with get_connection() as conn:
        with conn.cursor() as cursor:
            inserted = 0
            skipped = 0

            for nim, nama, prodi, angkatan in MAHASISWA:
                # Cek apakah sudah ada
                cursor.execute(
                    "SELECT nim FROM mahasiswa WHERE nim = %s", (nim,)
                )
                existing = cursor.fetchone()

                if existing:
                    print(f"  ⏭️  {nim} — {nama} (sudah ada)")
                    skipped += 1
                else:
                    cursor.execute(
                        "INSERT INTO mahasiswa (nim, nama, prodi, angkatan, status) VALUES (%s, %s, %s, %s, 'aktif')",
                        (nim, nama, prodi, angkatan)
                    )
                    print(f"  ✅ {nim} — {nama}")
                    inserted += 1

    print(f"\nSelesai! {inserted} ditambahkan, {skipped} sudah ada.")

    # Tampilkan semua mahasiswa
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT nim, nama, prodi, angkatan FROM mahasiswa ORDER BY nim")
            rows = cursor.fetchall()
            print(f"\n📋 Total mahasiswa: {len(rows)}")
            print(f"{'NIM':<15} {'Nama':<35} {'Prodi':<8} {'Angkatan'}")
            print("─" * 70)
            for r in rows:
                print(f"{r['nim']:<15} {r['nama']:<35} {r['prodi']:<8} {r['angkatan']}")


if __name__ == "__main__":
    seed()
