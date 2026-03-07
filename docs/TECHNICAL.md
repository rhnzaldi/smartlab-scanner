# 🔬 Dokumentasi Teknis — Smart-Lab SV IPB

> Dokumentasi ini menjelaskan arsitektur ML pipeline, backend API, dan database untuk proyek Smart-Lab SV IPB.

---

## 1. Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────┐
│                     main.py (FastAPI)                        │
│  REST: /api/scan  /api/face/*  /api/checkout  /api/status   │
│  WS:   /ws/scan                                             │
└────────────┬────────────────────────┬───────────────────────┘
             │                        │
     ┌───────▼───────┐       ┌───────▼───────┐
     │  ml/pipeline  │       │ ml/face_verify│
     │  (KTM Scan)   │       │ (InsightFace) │
     └───────┬───────┘       └───────────────┘
             │
   ┌─────────┼─────────┐
   │         │         │
┌──▼──┐  ┌──▼───┐  ┌──▼──────┐
│YOLO │  │OCR/QR│  │Validator│
│v8   │  │      │  │(Regex)  │
└─────┘  └──────┘  └─────────┘
   │         │         │
   └─────────┼─────────┘
             │
     ┌───────▼───────┐
     │ db/database   │
     │  (MySQL)      │
     └───────────────┘
```

---

## 2. ML Pipeline (`ml/pipeline.py`)

### Kelas: `KTMPipeline`

Pipeline utama yang mengorkestrasi seluruh proses scan KTM dalam 5 tahap:

```
Frame Kamera → YOLO Detect → Crop → Preprocess → Extract → Validate → ScanResult
```

#### Konstruktor

| Parameter | Default | Keterangan |
|-----------|---------|------------|
| `model_path` | `"models/best.pt"` | Path ke file model YOLOv8 |
| `confidence_threshold` | `0.35` | Minimum confidence YOLO |
| `iou_threshold` | `0.45` | Non-Max Suppression threshold |
| `device` | `"cpu"` | `"cpu"` atau `"cuda"` |

#### Method Utama

**`process_frame(frame: np.ndarray) → ScanResult`**

Proses satu frame melalui 5 tahap:

| Tahap | Apa yang Terjadi | File yang Terlibat |
|-------|------------------|--------------------|
| 1. **YOLO Detect** | Model YOLOv8 mendeteksi 4 area: `qr_code`, `text_nim`, `text_nama`, `foto` | `pipeline.py` |
| 2. **Crop** | Potong bounding box dari frame asli. Jika ada duplikat label, ambil confidence tertinggi | `pipeline.py` |
| 3. **Preprocess** | Grayscale → Blur → Sharpen (untuk OCR). OTSU threshold (untuk QR) | `preprocessor.py` |
| 4. **Extract** | PaddleOCR membaca teks NIM & Nama. pyzbar decode QR code | `extractor.py` |
| 5. **Validate** | Bersihkan OCR error (I→1, O→0). Double-check NIM QR vs NIM OCR | `validator.py` |

#### Dataclass: `ScanResult`

```python
@dataclass
class ScanResult:
    success: bool = False           # Ada data valid yang ditemukan?
    status: str = "no_detection"    # Status pipeline
    nim_qr: str | None              # NIM dari QR code
    nim_ocr: str | None             # NIM dari OCR teks
    nim_final: str | None           # NIM terpilih (QR prioritas)
    nama: str | None                # Nama dari OCR
    detections: List[Detection]     # Semua objek YOLO
    confidences: Dict[str, float]   # Confidence per label
    inference_time_ms: float        # Waktu YOLO inference
    total_time_ms: float            # Total waktu pipeline
```

#### YOLO Label Mapping

Model YOLO dilatih untuk mendeteksi 4 objek pada KTM IPB:

| Index | Label | Deskripsi |
|-------|-------|-----------|
| 0 | `foto` | Area foto mahasiswa |
| 1 | `qr_code` | QR Code di pojok kanan |
| 2 | `text_nama` | Area teks nama |
| 3 | `text_nim` | Area teks NIM |

---

## 3. Face Verification (`ml/face_verify.py`)

### Engine: InsightFace ArcFace (`buffalo_l`)

| Spesifikasi | Detail |
|-------------|--------|
| Model | ArcFace w600k_r50 (via InsightFace) |
| Embedding | 512-D float32 vector |
| Akurasi | 99.5%+ untuk wajah Asia |
| Kecepatan | ~50-80ms per frame (CPU) |
| Ukuran Model | ~300MB (download otomatis saat pertama kali) |
| Privasi | Hanya menyimpan 512 angka, **BUKAN foto wajah** |

### Kelas: `FaceVerifier`

| Method | Input | Output | Keterangan |
|--------|-------|--------|------------|
| `enroll(frame)` | BGR image | `{success, encoding, message}` | Ambil wajah terbesar → hasilkan 512-D embedding |
| `verify(frame)` | BGR image | `{verified, similarity, message}` | Compare wajah live vs referensi |
| `set_reference_from_encoding(emb)` | numpy array | `bool` | Load embedding dari DB sebagai referensi |
| `clear_reference()` | — | — | Reset internal state |

### Threshold

| Nilai | Arti |
|-------|------|
| `similarity ≥ 0.50` | ✅ Wajah **COCOK** (match) |
| `similarity < 0.50` | ❌ Wajah **DITOLAK** (reject) |

### Strategi Deteksi Wajah

Jika ada beberapa wajah dalam frame, InsightFace memilih **wajah terbesar** (asumsi: yang paling dekat ke kamera). Dihitung dari luas bounding box `(x2-x1) * (y2-y1)`.

---

## 4. Text & QR Extractor (`ml/extractor.py`)

### PaddleOCR v5

- **Lazy-loaded**: Hanya dimuat saat pertama kali dipanggil
- **Language**: `en` (teks KTM IPB dalam bahasa Inggris/Alfabet)
- **Minimum size**: Gambar < 16px tinggi akan di-skip
- **Auto-upscale**: Gambar < 64px tinggi otomatis di-resize

### pyzbar (QR Decoder)

Mencoba 4 strategi secara berurutan:

| Strategi | Teknik | Keterangan |
|----------|--------|------------|
| 1 | Raw image | Langsung decode |
| 2 | Grayscale | Hilangkan warna |
| 3 | OTSU binary | Kontras hitam-putih |
| 4 | Inverted | Kebalikan binary |

---

## 5. Image Preprocessor (`ml/preprocessor.py`)

Utility OpenCV untuk meningkatkan kualitas gambar sebelum OCR/QR decode.

### Fungsi yang Tersedia

| Fungsi | Input → Output | Keterangan |
|--------|---------------|------------|
| `to_grayscale(img)` | BGR → Grayscale | Skip jika sudah grayscale |
| `apply_blur(img, ksize=3)` | Image → Blurred | Gaussian blur, kurangi noise |
| `apply_sharpen(img)` | Image → Sharpened | Unsharp mask kernel 3x3 |
| `apply_threshold(img)` | Image → Binary | Adaptive Gaussian threshold |
| `resize_if_small(img, min=50)` | Image → Resized | Upscale jika < 50px tinggi |

### Chain Preprocessing

```
preprocess_for_ocr:  resize → grayscale → blur → sharpen
preprocess_for_qr:   resize(100px) → grayscale → OTSU threshold
```

---

## 6. Data Validator (`ml/validator.py`)

### `clean_nim(raw)` — Pembersih NIM dari OCR

OCR sering salah baca karakter. Fungsi ini mengoreksi:

| Karakter OCR | Koreksi |
|-------------|---------|
| `I`, `l`, `\|` | `1` |
| `O`, `o` | `0` |
| `S`, `s` | `5` |
| `Z`, `z` | `2` |
| `B` | `8` |
| `G` | `6` |
| `T` | `7` |

**Format NIM yang didukung:**
- `J0403231061` — J + 10 digit (format baru)
- `J3B121015` — J3B + 6 digit (format lama)

### `clean_name(raw)` — Pembersih Nama dari OCR

1. Hapus trailing NIM yang ikut terbawa (misal: `"RAIHAN ZALDI J040"` → `"Raihan Zaldi"`)
2. Hapus karakter non-alfabet
3. Title case
4. Skip jika < 2 karakter

### `validate_nim_match(nim_qr, nim_ocr)` — Double Validation

Membandingkan NIM dari QR code dengan NIM dari OCR teks:

| Kondisi | Hasil |
|---------|-------|
| Sama persis | ✅ Match |
| Beda 1-2 karakter | ⚠️ Partial match (OCR noise) |
| Beda > 2 karakter | ❌ Mismatch |
| Salah satu kosong | ⚠️ Warning |

---

## 7. Database (`db/database.py`)

### Engine: MySQL (`smartlab_db` @ localhost)

### Tabel: `mahasiswa`

| Kolom | Tipe | Keterangan |
|-------|------|------------|
| `nim` | TEXT PRIMARY KEY | NIM mahasiswa |
| `nama` | TEXT | Nama lengkap |
| `prodi` | TEXT | Program studi |
| `angkatan` | INTEGER | Tahun masuk |
| `status` | TEXT | Status mahasiswa |
| `face_encoding` | LONGBLOB | 512-D float32 (2048 bytes) |

### Tabel: `peminjaman`

| Kolom | Tipe | Keterangan |
|-------|------|------------|
| `id` | INT AUTO_INCREMENT PRIMARY KEY | Auto-increment |
| `nim` | TEXT | FK → mahasiswa |
| `lab` | TEXT | Nama lab |
| `waktu_masuk` | TEXT | Timestamp check-in |
| `waktu_keluar` | TEXT | Timestamp check-out (NULL = masih di lab) |
| `status` | TEXT | `"aktif"` atau `"selesai"` |

### Fungsi Database

| Fungsi | Keterangan |
|--------|------------|
| `init_db()` | Buat tabel + seed data awal |
| `verify_student(nim, nama)` | Lookup NIM + fuzzy match nama (≥50%) |
| `check_in(nim)` | Catat masuk lab. Cek duplikat aktif |
| `check_out(nim)` | Catat keluar. Hitung durasi |
| `get_active_peminjaman()` | List semua yang masih di lab |
| `save_face_encoding(nim, enc)` | Simpan 512-D embedding ke BLOB |
| `load_face_encoding(nim)` | Load embedding dari BLOB |
| `delete_face_encoding(nim)` | Reset encoding ke NULL |

### Safety Features

- **Context Manager** (`get_connection()`): Auto commit, rollback on error, always close
- **Operation Timer** (`@_timed_db_op`): Log durasi setiap operasi DB
- **Name Matching** (`fuzzy_name_match()`): `SequenceMatcher` dengan threshold 50%

---

## 8. Backend API (`main.py`)

### Security Tags

| Tag | Deskripsi | Lokasi |
|-----|-----------|--------|
| `[S1]` | CORS restrictive | Line 55-59 |
| `[S2]` | Input validation (NIM regex) | Line 155-166 |
| `[S3]` | File size limit 10MB | Line 51-52 |
| `[S4]` | No error detail leak | Line 204 |
| `[S5]` | Admin API Key auth | Line 61-75 |

### Singleton Pattern

| Object | Lazy Load? | Lock? |
|--------|-----------|-------|
| `KTMPipeline` | Saat startup (`lifespan()`) | Tidak perlu (stateless) |
| `FaceVerifier` | Saat request pertama | ✅ `asyncio.Lock()` |

### Environment Variables

| Variable | Default | Keterangan |
|----------|---------|------------|
| `CORS_ORIGINS` | `localhost:3000,5173` | Daftar origin yang diizinkan |
| `ADMIN_API_KEY` | `CHANGE-ME-IN-PRODUCTION` | API key untuk endpoint admin |

---

## 9. Struktur File Proyek

```
ScanKtm/
├── main.py                 # FastAPI server (8 endpoints + 1 WebSocket)
├── test_api.py             # Pytest unit tests (8 tests)
├── test_webcam.py          # Standalone webcam tester (6-phase state machine)
├── seed_mahasiswa.py       # Script seed data mahasiswa ke DB
├── evaluate_models.py      # Script evaluasi akurasi ML
├── augment_dataset.py      # Script augmentasi dataset training
├── requirements.txt        # Python dependencies
├── models/
│   └── best.pt             # YOLOv8 trained model (tidak di-commit)
├── ml/
│   ├── __init__.py
│   ├── pipeline.py         # Orchestrator: YOLO → OCR → Validate
│   ├── face_verify.py      # InsightFace ArcFace enrollment + verify
│   ├── extractor.py        # PaddleOCR + pyzbar
│   ├── preprocessor.py     # OpenCV image preprocessing
│   └── validator.py        # Regex cleaning + NIM double-validation
├── db/
│   ├── __init__.py
│   └── database.py         # MySQL (PyMySQL) CRUD + face encoding storage
└── docs/
    ├── FRONTEND_INTEGRATION.md  # Panduan integrasi untuk tim Frontend
    └── TECHNICAL.md             # Dokumen ini
```
