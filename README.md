# 🎓 Smart-Lab SV IPB — KTM Scanner

Sistem pemindaian **Kartu Tanda Mahasiswa (KTM)** secara real-time menggunakan Computer Vision + Face Verification untuk manajemen akses laboratorium.

## ✨ Fitur

- 🔍 **YOLO Detection** — Deteksi QR Code, NIM, Nama, dan Pas Foto pada KTM
- 📝 **PaddleOCR** — Ekstraksi teks NIM dan Nama dari gambar
- 📱 **QR Decode** — Baca NIM dari QR Code (pyzbar) dengan 4 strategi fallback
- ✅ **Double Validation** — Cross-check NIM dari QR dan OCR
- 👤 **Face Verification** — Bandingkan wajah live dengan foto KTM (OpenCV Haar Cascade)
- 🗄️ **Database** — Verifikasi identitas dan pencatatan check-in/check-out lab
- 🌐 **REST + WebSocket API** — FastAPI backend untuk integrasi frontend

## 🏗 Arsitektur

```
Webcam Frame
     │
     ▼
┌──────────┐    ┌──────────────┐    ┌───────────┐    ┌────────────┐
│  YOLOv8  │───▶│  Preprocess  │───▶│ PaddleOCR │───▶│  Validator  │
│  Detect  │    │  (OpenCV)    │    │ + pyzbar  │    │  (Regex)    │
└──────────┘    └──────────────┘    └───────────┘    └─────┬──────┘
   4 class:        grayscale          NIM text              │
   qr_code         blur              Nama text        ┌────▼──────┐
   text_nim         sharpen           QR data          │  SQLite   │
   text_nama        threshold                          │ Verify +  │
   face_photo                                          │ Check-in  │
                                                       └────┬──────┘
                                                            │
                                                       ┌────▼──────┐
                                                       │   Face    │
                                                       │  Verify   │
                                                       └───────────┘
```

### 🔄 Webcam Scanner — 5-Phase State Machine

```
┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐    ┌──────────┐
│ SCANNING │───▶│  IDENTITAS   │───▶│  VERIFIKASI  │───▶│ COMPLETE │───▶│ COOLDOWN │
│          │    │  DITEMUKAN   │    │    WAJAH     │    │          │    │          │
│ Deteksi  │    │   (2 detik)  │    │  (5 detik)   │    │ (2 detik)│    │ (3 detik)│
│ KTM      │    │  Info DB     │    │  Countdown   │    │ Berhasil │    │  Idle    │
└──────────┘    └──────────────┘    └──────────────┘    └──────────┘    └──────────┘
      ▲                                                                      │
      └──────────────────────────────────────────────────────────────────────┘
```

## 📋 Prerequisites

- **Python** 3.9+
- **Webcam** (built-in / USB)
- **Model** `best.pt` — YOLOv8 yang sudah di-train (lihat [Training](#-training))
- **OS**: macOS / Linux (belum ditest di Windows)

## 🚀 Quick Start

```bash
# 1. Clone repo
git clone https://github.com/rhnzaldi/smartlab-scanner.git
cd smartlab-scanner

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy trained model
cp /path/to/your/best.pt models/best.pt

# 4. Seed database (opsional — edit daftar mahasiswa di file ini)
python seed_mahasiswa.py

# 5. Jalankan webcam scanner
python test_webcam.py

# 6. Atau jalankan API server
python main.py
```

### macOS — Jika ada error library:

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/lib python test_webcam.py
```

## 🎮 Webcam Controls

| Key | Fungsi |
|-----|--------|
| `q` | Quit |
| `s` | Save frame ke `captures/` |
| `SPACE` | Pause / Resume |
| `+` / `-` | Adjust confidence threshold |
| `o` | Force trigger OCR |
| `c` | Manual check-out (input NIM) |
| `r` | Reset semua peminjaman aktif |

### Opsi Command Line

```bash
# Tanpa face verification (lebih cepat)
python test_webcam.py --no-face-verify

# Dengan face verification (default)
python test_webcam.py
```

### POST `/api/scan` — Contoh Response

```json
{
  "success": true,
  "status": "validated",
  "nim_final": "J0403231XXX",
  "nama": "Nama Mahasiswa",
  "nim_match": true,
  "db_verified": true,
  "db_prodi": "TPL",
  "checkin": {
    "success": true,
    "message": "📥 Check-in berhasil!"
  },
  "inference_time_ms": 45.2,
  "total_time_ms": 230.5
}
```

### WebSocket `/ws/scan`

```javascript
// Client kirim base64 image
ws.send(JSON.stringify({ image: "data:image/jpeg;base64,..." }))

// Server merespons dengan hasil scan + DB verify + check-in
```

## 🗄️ Database

Menggunakan **SQLite** (file: `smartlab.db`, auto-created).

### Tabel `mahasiswa`
| Kolom | Tipe | Keterangan |
|-------|------|------------|
| `nim` | TEXT (PK) | NIM mahasiswa |
| `nama` | TEXT | Nama lengkap |
| `prodi` | TEXT | Program studi |
| `angkatan` | INTEGER | Tahun masuk |
| `status` | TEXT | `aktif` / `nonaktif` |

### Tabel `peminjaman`
| Kolom | Tipe | Keterangan |
|-------|------|------------|
| `id` | INTEGER (PK) | Auto increment |
| `nim` | TEXT (FK) | Referensi ke mahasiswa |
| `lab` | TEXT | Nama lab |
| `waktu_masuk` | TIMESTAMP | Waktu check-in |
| `waktu_keluar` | TIMESTAMP | Waktu check-out |
| `status` | TEXT | `aktif` / `selesai` |

### Menambah Data Mahasiswa

Edit file `seed_mahasiswa.py`:

```python
MAHASISWA = [
    ("J04032310XX", "Nama Mahasiswa", "Prodi", 2023),
    # Tambahkan baris baru di sini
]
```

Lalu jalankan:

```bash
python seed_mahasiswa.py
```

## 🤖 Training

Model YOLOv8 di-train menggunakan Google Colab. Lihat `train_colab.py` untuk panduan lengkap.

### Dataset
- **Objek**: `qr_code`, `text_nim`, `text_nama`, `face_photo`
- **Augmentasi**: `augment_dataset.py` — rotasi, blur, brightness, noise
- **Platform**: Roboflow untuk labeling + export

### Langkah Training
1. Upload dataset ke Roboflow
2. Buka Google Colab
3. Jalankan `train_colab.py` (ikuti instruksi di file)
4. Download `best.pt` ke `models/best.pt`

## 📁 Struktur Proyek

```
smartlab-scanner/
├── ml/                        # ML Pipeline
│   ├── pipeline.py            # Orchestrator: detect → crop → extract → validate
│   ├── preprocessor.py        # OpenCV preprocessing (grayscale, blur, sharpen)
│   ├── extractor.py           # PaddleOCR + pyzbar extraction
│   ├── validator.py           # Regex cleaning & NIM double-validation
│   └── face_verify.py         # Haar Cascade face verification
├── db/
│   └── database.py            # SQLite: mahasiswa + peminjaman management
├── models/
│   └── best.pt                # YOLOv8 model weights (git-ignored)
├── main.py                    # FastAPI REST + WebSocket server
├── test_webcam.py             # Standalone webcam scanner (5-phase UX)
├── seed_mahasiswa.py          # Script untuk seed data mahasiswa
├── train_colab.py             # Panduan training di Google Colab
├── augment_dataset.py         # Data augmentation script
├── requirements.txt           # Python dependencies
└── .gitignore
```

## ⚙️ Environment Variables

| Variable | Default | Keterangan |
|----------|---------|------------|
| `CORS_ORIGINS` | `http://localhost:3000,...` | Allowed CORS origins (comma-separated) |
| `DYLD_LIBRARY_PATH` | — | macOS: set ke `/opt/homebrew/lib` jika perlu |

## 📝 Tech Stack

| Komponen | Teknologi |
|----------|-----------|
| Object Detection | YOLOv8 (Ultralytics) |
| OCR | PaddleOCR v5 |
| QR Decode | pyzbar |
| Face Detection | OpenCV Haar Cascade |
| Face Matching | Histogram Comparison |
| Backend | FastAPI + Uvicorn |
| Database | SQLite (WAL mode) |
| Image Processing | OpenCV + NumPy |

## 📄 License

MIT
