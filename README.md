# 🎓 Smart-Lab SV IPB — AI Backend Scanner

Sistem backend cerdas untuk pemindaian **Kartu Tanda Mahasiswa (KTM)** secara *real-time* menggunakan Computer Vision dan Verifikasi Wajah untuk otomatisasi akses laboratorium kampus.

---

## ✨ Fitur Utama

- 🔍 **YOLO Object Detection** — Deteksi wilayah QR Code, NIM, Nama, dan Pas Foto pada KTM secara instan.
- 📝 **OCR & QR Extraction** — Ekstraksi data NIM dan Nama menggunakan PaddleOCR v5 dan ekstraksi QR Code dengan algoritma multi-stage (pyzbar).
- ✅ **Double Validation** — Sistem membandingkan NIM hasil bacaan OCR dengan NIM hasil bacaan QR Code untuk akurasi ekstra.
- 👤 **Face Biometric Validation** — Pendaftaran (Enrollment) dan verifikasi wajah mahasiswa harian menggunakan InsightFace ArcFace (99.83% akurasi).
- 🔐 **Privacy-First Design** — Sistem **tidak menyimpan** foto wajah mahasiswa di database, melainkan hanya menyimpan *512-D float32 mathematical embedding* (~2KB), yang membuat data biometrik mustahil direkonstruksi menjadi wajah asli.
- 🌐 **REST + WebSocket API** — Menyediakan endpoint HTTP mutakhir untuk manajemen data dan koneksi WebSocket untuk *real-time continuous scanning*.

---

## 🏗 Arsitektur Machine Learning (ML Pipeline)

Setiap frame gambar atau foto yang dikirimkan oleh Frontend diproses melalui 5 tahap pipelining berikut:

```
Frame HTTP/WS 
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

## 📋 Persyaratan Sistem

- **Python** 3.9, 3.10, 3.11, atau 3.12 (Tidak disarankan <= 3.8)
- **Kamera** (Built-in / Eksternal USB) yang terhubung dengan akses internet Frontend.
- Memori RAM minimal 2GB kosong untuk memuat model pendeteksi objek dan wajah.
- OS yang didukung: Linux, macOS, dan Windows. (Linux/macOS lebih diprioritaskan untuk server deployment).

## 🚀 Menjalankan Server Lokal (Quick Start)

### 1. Kloning Repositori

```bash
git clone https://github.com/rhnzaldi/smartlab-scanner.git
cd smartlab-scanner
```

### 2. Buat Virtual Environment (Sangat Disarankan)

Agar tidak bentrok dengan library proyek lain, gunakan Virtual Environment.

**Untuk Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**Untuk macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalasi Dependencies

Pastikan tulisan `(venv)` sudah muncul di terminal Anda, lalu jalankan:

```bash
pip install -r requirements.txt
```

**Opsional — Akselerasi GPU (Hanya Windows/Linux dengan NVIDIA GPU):**

Ada dua cara:

**Cara 1 (Disarankan): Gunakan branch `windows-gpu`**

Branch ini sudah mengubah semua konfigurasi ke mode GPU secara otomatis (`onnxruntime-gpu`, `paddlepaddle-gpu`, CUDA provider di InsightFace dan PaddleOCR):

```bash
git checkout windows-gpu
pip install -r requirements.txt
```

**Cara 2 (Manual): Tetap di branch `main`, ganti library satu per satu**

```bash
# Pastikan sudah install CUDA Toolkit 11.8 atau 12.x dari https://developer.nvidia.com/cuda-downloads
pip uninstall onnxruntime -y
pip install onnxruntime-gpu
```

> Jika tidak memiliki NVIDIA GPU, **tetap di branch `main`** dan tidak perlu melakukan langkah ini. Sistem akan berjalan normal menggunakan CPU.

**Khusus Windows — Fix pyzbar (QR Code reader):**

Jika muncul error `FileNotFoundError: Could not find module 'libzbar-64.dll'`, install Visual C++ Redistributable:

```bash
# Opsi 1: Install via pip
pip install pyzbar[scripts]

# Opsi 2: Download Visual C++ Redistributable dari Microsoft
# https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist
```

### 4. Masukkan Model Trained YOLO

Pastikan Anda memiliki file model YOLOv8 kustom Anda (`best.pt`) dan masukkan ke dalam direktori `models/`:

```bash
# Windows:
mkdir models
copy \jalur\ke\best.pt models\best.pt

# macOS/Linux:
mkdir -p models
cp /jalur/ke/best.pt models/best.pt
```

### 5. Isi Data Mahasiswa Awal

Edit file `seed_mahasiswa.py` untuk memasukkan daftar NIM dan nama mahasiswa, lalu jalankan:

```bash
python seed_mahasiswa.py
```

> Langkah ini hanya perlu dilakukan **sekali** saat pertama kali setup. Database `smartlab.db` akan dibuat otomatis.

### 6. Mulai Server API

Jalankan FastAPI backend menggunakan server uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Jika server sudah berjalan, buka **`http://localhost:8000/docs`** 

*Catatan: Konfigurasi CORS hanya akan mengizinkan request API dari alamat Frontend spesifik seperti localhost:3000.*

---

## 📡 Daftar REST API (Endpoint)

Integrasi Frontend dengan Server ini akan membutuhkan endpoint berikut:

| Method | Endpoint | Deskripsi | Authentication |
|--------|----------|-----------|----------------|
| `GET` | `/health` | Memeriksa apakah Backend ML menyala | Terbuka |
| `POST` | `/api/scan` | Upload base64/blob foto KTM untuk diperiksa AI | Terbuka |
| `POST` | `/api/face/enroll` | Mendaftarkan pemetaan 512-D wajah baru (Awal Semester) | Terbuka |
| `POST` | `/api/face/verify` | Memverifikasi identitas pengguna lab harian | Terbuka |
| `POST` | `/api/checkout/{nim}` | Merekam waktu keluar mahasiswa (Check-out) | Terbuka |
| `DELETE` | `/api/face/{nim}` | [ADMIN] Menghapus data biometrik wajah. | `X-Admin-Key` |
| `GET` | `/api/status` | [ADMIN] Daftar mahasiswa yang hadir di lab. | `X-Admin-Key` |

> Panduan integrasi spesifik untuk tim pengembang web Frontend tersedia di file `/docs/FRONTEND_INTEGRATION.md`.

---

## 🛠️ Tech Stack & Model Library

| Komponen | Teknologi Implementasi | Spesifikasi / Ukuran |
|----------|-----------|--------|
| Object Detection | **YOLOv8n** (Ultralytics) | 6MB, 4 label kelas, ~48ms inference per frame |
| Ekstraksi OCR Teks | **PaddleOCR v5** | ~500ms/image (Deteksi Model Bahasa Inggris/Alfabet) |
| Decoder QR Code | **pyzbar** | Dengan OpenCV 4 fallback filter strategy. |
| Pengenalan Wajah AI | **InsightFace ArcFace** | `buffalo_l`, 512-Dimensi, ONNX Runtime (~110ms /frame) |
| Web Backend | **FastAPI** + **Uvicorn** | Python asinkron, ASGI |
| Manajemen Database | **SQLite** (lokal) | Native, file DB auto-generated (`smartlab.db`) |
| Pengolahan Gambar Matrix | **OpenCV** + **NumPy** | Grayscale, blur, ROI cropping, adaptive array |

---

## ❓ Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `ModuleNotFoundError: No module named 'paddleocr'` | Pastikan sudah `pip install -r requirements.txt` di dalam venv |
| `libzbar-64.dll not found` (Windows) | Install Visual C++ Redistributable atau `pip install pyzbar[scripts]` |
| `CUDA out of memory` | Kurangi `det_size` di `face_verify.py` dari `(640,640)` ke `(320,320)` |
| Server port 8000 sudah dipakai | Jalankan di port lain: `uvicorn main:app --port 8001` |
| Model YOLO tidak ditemukan | Pastikan file `best.pt` ada di folder `models/` |
| InsightFace download lambat | Model `buffalo_l` (~300MB) di-download otomatis saat pertama kali. Tunggu selesai |

---

