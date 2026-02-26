# 🎓 Smart-Lab SV IPB — KTM Scanner

Sistem pemindaian **Kartu Tanda Mahasiswa (KTM)** secara real-time menggunakan Computer Vision + Face Verification untuk manajemen akses laboratorium.

## ✨ Fitur

- 🔍 **YOLO Detection** — Deteksi QR Code, NIM, Nama, dan Pas Foto pada KTM
- 📝 **PaddleOCR v5** — Ekstraksi teks NIM dan Nama dari gambar
- 📱 **QR Decode** — Baca NIM dari QR Code (pyzbar) dengan 4 strategi fallback
- ✅ **Double Validation** — Cross-check NIM dari QR dan OCR
- 👤 **Face Verification** — InsightFace ArcFace (99.83% akurasi LFW)
- 🔐 **Face Enrollment** — Registrasi wajah pertama kali via webcam → 512-D embedding di DB
- 🗄️ **Database** — Verifikasi identitas dan pencatatan check-in/check-out lab
- 🌐 **REST + WebSocket API** — FastAPI backend untuk integrasi frontend
- 🔒 **Privacy** — Hanya menyimpan 512 angka encoding, bukan foto wajah

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
                                                  ┌─────────▼──────────┐
                                                  │  InsightFace       │
                                                  │  ArcFace (512-D)   │
                                                  │  Enroll / Verify   │
                                                  └────────────────────┘
```

### 🔄 Webcam Scanner — 6-Phase State Machine

```
┌──────────┐   ┌───────────┐   ┌────────────┐   ┌──────────┐   ┌──────────┐
│ SCANNING │──▶│ IDENTITAS │──▶│ REGISTRASI │──▶│ COMPLETE │──▶│ COOLDOWN │
│          │   │ DITEMUKAN │   │   WAJAH    │   │          │   │          │
│  YOLO +  │   │  (3 dtk)  │   │  (10 dtk)  │   │ (2 dtk)  │   │ (3 dtk)  │
│   OCR    │   │  Info DB  │   │ Pertama 1x │   │ Berhasil │   │   Idle   │
└──────────┘   └───────────┘   └────────────┘   └──────────┘   └──────────┘
      ▲                              │                               │
      │                   ┌──────────┘ (sudah terdaftar)             │
      │                   ▼                                          │
      │            ┌────────────┐                                    │
      │            │ VERIFIKASI │                                    │
      │            │   WAJAH    │─── gagal ──▶ COOLDOWN (DITOLAK)   │
      │            │  (7 dtk)   │                                    │
      │            └────────────┘                                    │
      └──────────────────────────────────────────────────────────────┘
```

**Flow pertama kali:** Scan KTM → DB match → **Registrasi Wajah** (kuning) → Check-in ✅
**Flow selanjutnya:** Scan KTM → DB match → **Verifikasi Wajah** (biru) → Cocok → Check-in ✅
**Wajah tidak cocok:** → ❌ **DITOLAK** → Cooldown (tidak ada check-in)

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
| `c` | Manual check-out |
| `r` | Reset semua peminjaman aktif |
| `f` | **Reset face encoding** (test ulang enrollment) |

### Opsi Command Line

```bash
python test_webcam.py                  # Default (face verify ON)
python test_webcam.py --no-face-verify # Tanpa face verification
python test_webcam.py --camera 1       # Gunakan kamera kedua
python test_webcam.py --confidence 0.5 # Threshold YOLO 50%
```

## 👤 Face Enrollment & Verification

### Cara Kerja

1. **Enrollment (pertama kali)**:
   - Scan KTM → identitas ditemukan di DB
   - Sistem mendeteksi belum ada encoding wajah → masuk mode **REGISTRASI**
   - Hadap ke kamera → InsightFace menangkap wajah → 512-D embedding disimpan di DB
   - Check-in berhasil ✅

2. **Verification (selanjutnya)**:
   - Scan KTM → identitas ditemukan di DB
   - Encoding wajah sudah ada → masuk mode **VERIFIKASI**
   - Hadap ke kamera → InsightFace membandingkan wajah live vs encoding di DB
   - Cocok (≥50%) → Check-in ✅ | Tidak cocok → ❌ DITOLAK

### Privacy & Security

- **Hanya encoding yang disimpan** — 512 angka float32 (~2KB per mahasiswa)
- **Bukan foto** — encoding tidak bisa di-reverse menjadi gambar wajah
- **One-way** — aman dari pencurian data biometrik

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
| `face_encoding` | BLOB | 512-D face embedding (float32) |

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
3. Jalankan cell-cell di `train_colab.py` (CELL 1 - CELL 7)
4. Download `best.pt` ke `models/best.pt`

### Evaluasi Model
```bash
# Jalankan lokal — benchmark semua model ML
python evaluate_models.py
```

Di Google Colab (untuk YOLO metrics):
- **CELL 5**: Precision, Recall, F1 Score, mAP@50, mAP@50-95 (per-class)
- **CELL 5B**: Confusion matrix + training curves + download ZIP
- **CELL 5C**: Tabel Markdown siap copy-paste ke laporan

## 📁 Struktur Proyek

```
smartlab-scanner/
├── ml/                        # ML Pipeline
│   ├── pipeline.py            # Orchestrator: detect → crop → extract → validate
│   ├── preprocessor.py        # OpenCV preprocessing (grayscale, blur, sharpen)
│   ├── extractor.py           # PaddleOCR + pyzbar extraction
│   ├── validator.py           # Regex cleaning & NIM double-validation
│   └── face_verify.py         # InsightFace ArcFace face enrollment & verification
├── db/
│   └── database.py            # SQLite: mahasiswa + peminjaman + face encoding
├── models/
│   └── best.pt                # YOLOv8 model weights (git-ignored)
├── main.py                    # FastAPI REST + WebSocket server
├── test_webcam.py             # Standalone webcam scanner (6-phase UX)
├── evaluate_models.py         # Benchmark semua model ML (untuk laporan)
├── seed_mahasiswa.py          # Script untuk seed data mahasiswa
├── train_colab.py             # Panduan training + evaluasi di Google Colab
├── augment_dataset.py         # Data augmentation script
├── requirements.txt           # Python dependencies
└── .gitignore
```

## 📊 Tech Stack

| Komponen | Teknologi | Detail |
|----------|-----------|--------|
| Object Detection | YOLOv8n (Ultralytics) | 6MB, 4 kelas, ~48ms/frame |
| OCR | PaddleOCR v5 | English mode, ~500ms/image |
| QR Decode | pyzbar | 4 strategi fallback |
| Face Recognition | InsightFace ArcFace (buffalo_l) | 512-D, 99.83% LFW, ~110ms/frame |
| Backend | FastAPI + Uvicorn | REST + WebSocket |
| Database | SQLite (WAL mode) | Auto-created, zero config |
| Image Processing | OpenCV + NumPy | Preprocessing pipeline |
| ML Runtime | ONNX Runtime | CPU optimized |

## ⚙️ Environment Variables

| Variable | Default | Keterangan |
|----------|---------|------------|
| `CORS_ORIGINS` | `http://localhost:3000,...` | Allowed CORS origins (comma-separated) |
| `DYLD_LIBRARY_PATH` | — | macOS: set ke `/opt/homebrew/lib` jika perlu |

## 📄 License

MIT
