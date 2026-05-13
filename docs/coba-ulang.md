# Panduan Jalanin ScanKtm di Windows (Branch coba-low-spec)
> Baca dari atas ke bawah, jangan di-skip

---

## ⚠️ Perlu Tahu Dulu

Proyek ini punya **2 bagian** yang harus jalan **bersamaan tapi di terminal berbeda**:

| Bagian | Apa | Jalan di |
|---|---|---|
| **Backend API** | Server FastAPI (database, login, dll) | Terminal 1 |
| **Scanner ML** | Kamera + YOLO + OCR + Face | Terminal 2 |

Keduanya **harus jalan di laptop/PC yang ada di ruang lab** — tidak bisa di-hosting ke internet
karena Scanner ML butuh kamera fisik yang terhubung langsung.

---

## Langkah 0 — Syarat Awal

Pastikan sudah terinstall:
- **Python 3.10 atau 3.11** (jangan 3.12, banyak library belum support)
  - Cek: `python --version`
- **MySQL** sudah jalan dan sudah buat database `smartlab_db`
- **Git**
- File **`models/best.pt`** — minta ke Zaldi, taruh di folder `ScanKtm/models/`
- File **`.env`** — minta ke Zaldi, taruh di root folder `ScanKtm/`

---

## Langkah 1 — Clone dari GitHub

Buka PowerShell, lalu:

```powershell
git clone -b coba-low-spec https://github.com/rhnzaldi/smartlab-scanner.git ScanKtm
cd ScanKtm
```

> Kalau sudah pernah clone sebelumnya, cukup update:
> ```powershell
> cd ScanKtm
> git fetch --all
> git checkout coba-low-spec
> git pull origin coba-low-spec
> ```

---

## Langkah 2 — Buat Virtual Environment

```powershell
python -m venv venv
```

> Hanya dilakukan **sekali saja** saat pertama kali setup.

---

## Langkah 3 — Install Semua Library

Aktifkan venv dulu, lalu install:

```powershell
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> ⏳ Proses ini lama (~10-30 menit) karena download PaddleOCR + InsightFace + YOLO.
> Pastikan koneksi internet stabil.

### Kalau pyzbar error:
```powershell
pip install pyzbar[scripts]
```

---

## Langkah 4 — (Opsional) Aktifkan GPU

> Lewati langkah ini dulu, jalankan pakai CPU biasa. GPU hanya untuk performa lebih baik.

**Kalau laptop punya AMD GPU atau Intel Iris Xe:**
```powershell
pip uninstall onnxruntime -y
pip install onnxruntime-directml
```

**Kalau DirectML malah error/crash, kembalikan ke CPU:**
```powershell
pip uninstall onnxruntime-directml -y
pip install onnxruntime
```

---

## Langkah 5 — Setup Database (Sekali Saja)

Pastikan MySQL sudah jalan, lalu jalankan seeder:

```powershell
.\venv\Scripts\activate
python seed_mahasiswa.py --file students.xlsx
```

> File `students.xlsx` minta ke Zaldi.
> Seeder ini aman dijalankan berkali-kali — tidak akan duplikat data.

---

## Langkah 6 — Jalankan Backend (Terminal 1)

Buka **PowerShell baru** (Terminal 1):

```powershell
cd ScanKtm
.\venv\Scripts\activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Kalau berhasil, akan muncul:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

> Biarkan terminal ini tetap terbuka. Jangan ditutup.

---

## Langkah 7 — Jalankan Scanner ML (Terminal 2)

Buka **PowerShell baru lagi** (Terminal 2):

```powershell
cd ScanKtm
.\venv\Scripts\activate
python test_webcam.py --low-spec
```

> Flag `--low-spec` WAJIB dipakai di laptop tanpa GPU / spek rendah.
> Tanpa flag ini, program tetap bisa jalan tapi akan lebih berat.

Kalau kamera bukan di index 0 (tidak muncul gambar), coba:
```powershell
python test_webcam.py --low-spec --camera 1
```

---

## Setiap Kali Mau Jalan Ulang

Setiap membuka laptop baru / setelah restart, ulangi **Langkah 6 dan 7** saja:

```powershell
# Terminal 1 — Backend
cd ScanKtm
.\venv\Scripts\activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Scanner ML
cd ScanKtm
.\venv\Scripts\activate
python test_webcam.py --low-spec
```

---

## Tombol di Jendela Scanner

| Tombol | Fungsi |
|---|---|
| `q` | Keluar |
| `s` | Simpan screenshot |
| `SPACE` | Pause/resume |
| `r` | Reset peminjaman aktif |
| `f` | Reset data wajah |
| `c` | Check-out manual |