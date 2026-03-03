# 🎓 Smart-Lab SV IPB — AI Backend Scanner

Sistem backend cerdas untuk pemindaian **Kartu Tanda Mahasiswa (KTM)** secara *real-time* menggunakan Computer Vision dan Verifikasi Wajah (Biometric) untuk otomatisasi akses laboratorium kampus.

Sistem ini didesain sebagai backend independen (`FastAPI`) yang dapat diintegrasikan dengan frontend web modern (React/Next.js/Vue) melalui antarmuka REST API dan WebSocket.

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

### 2. Instalasi Variabel Lingkungan & Dependencies

Disarankan menggunakan `Virtual Environment` seperti `venv` atau `conda`.

```bash
pip install -r requirements.txt
```

### 3. Masukkan Model Trained YOLO

Pastikan Anda memiliki file model YOLOv8 kustom Anda (`best.pt`) dan masukkan ke dalam direktori `models/`:

```bash
mkdir -p models
# Copy model YOLOv8 Anda yang telah di-train:
# cp /jalur/ke/best.pt models/best.pt
```

### 4. Mulai Server API

Jalankan FastAPI backend menggunakan server uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Jika server sudah berjalan, buka **`http://localhost:8000/docs`** di browser komputer Anda untuk melihat **Swagger UI** interaktif yang menguraikan seluruh Endpoint HTTP backend. 

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

## 📄 Lisensi


