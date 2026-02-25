# 🎓 Smart-Lab SV IPB — ML Engine

Sistem pemindaian **Kartu Tanda Mahasiswa (KTM)** secara real-time menggunakan Computer Vision.

## 🏗 Arsitektur

```
Frame Webcam
    │
    ▼
┌──────────┐    ┌──────────────┐    ┌──────────┐    ┌────────────┐
│  YOLOv8  │───▶│  OpenCV Pre  │───▶│ PaddleOCR │───▶│  Validator │
│  Detect  │    │  processing  │    │ + pyzbar  │    │  (Regex)   │
└──────────┘    └──────────────┘    └──────────┘    └────────────┘
   4 objek:        grayscale         NIM teks         clean NIM
   qr_code         blur              Nama teks        clean Name
   nim_teks        sharpen            QR data          NIM match
   nama_teks       threshold
   pas_foto
```

## 📋 Prerequisites

- Python 3.9+
- Webcam (built-in / USB)
- File `best.pt` (trained YOLOv8 model) — lihat bagian Training

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy model ke folder models/
cp /path/to/your/best.pt models/best.pt

# 3. Test dengan webcam (tanpa server)
python test_webcam.py

# 4. Jalankan FastAPI server
python main.py
# atau: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 🎮 Webcam Test Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save frame ke `captures/` |
| `SPACE` | Pause / Resume |
| `+` / `-` | Adjust confidence threshold |

## 🤖 Training Model (Google Colab)

Lihat file `train_colab.py` untuk panduan lengkap training YOLOv8 di Google Colab menggunakan dataset Roboflow.

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/scan` | Upload image, get scan result |
| WS | `/ws/scan` | WebSocket real-time scanning |

### WebSocket Protocol

```javascript
// Client sends base64 image:
ws.send(JSON.stringify({ image: "data:image/jpeg;base64,..." }))

// Server responds:
{
  "success": true,
  "status": "validated",
  "nim_qr": "J3B121015",
  "nim_ocr": "J3B121015",
  "nim_final": "J3B121015",
  "nama": "Ahmad Zaldi",
  "nim_match": true,
  "validation_detail": "✅ NIM MATCH: 'J3B121015'",
  "inference_time_ms": 45.2,
  "total_time_ms": 230.5
}
```

## 📁 Struktur Proyek

```
ScanKtm/
├── ml/                    # ML Pipeline modules
│   ├── __init__.py
│   ├── pipeline.py        # KTMPipeline orchestrator
│   ├── preprocessor.py    # OpenCV preprocessing
│   ├── extractor.py       # PaddleOCR + pyzbar
│   └── validator.py       # Regex cleaning & validation
├── models/
│   └── best.pt            # YOLOv8 model (anda copy ke sini)
├── main.py                # FastAPI backend
├── test_webcam.py         # Standalone webcam test
├── train_colab.py         # Colab training guide
└── requirements.txt
```
