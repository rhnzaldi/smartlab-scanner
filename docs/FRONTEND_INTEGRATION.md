# 📘 Panduan Integrasi Frontend — Smart-Lab SV IPB

> Dokumen ini ditujukan untuk tim Frontend yang akan mengintegrasikan antarmuka web dengan Backend ML Engine.

---

## 1. Setup & Menjalankan Backend

```bash
# Clone repo backend
git clone https://github.com/rhnzaldi/smartlab-scanner.git
cd smartlab-scanner

# Install dependencies menggunakan Virtual Environment
# Windows:
python -m venv venv
.\venv\Scripts\activate
# macOS/Linux:
# python3 -m venv venv && source venv/bin/activate

pip install -r requirements.txt

# Jalankan server (port 8000)
# Pastikan terminal masih berada dalam state (venv)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Setelah jalan, buka **http://localhost:8000/docs** untuk melihat Swagger UI (dokumentasi interaktif).

> **CORS yang diizinkan:** `localhost:3000`, `localhost:5173`, `127.0.0.1:3000`  
> Jika Frontend jalan di port lain, minta ke tim Backend untuk menambahkan via env var `CORS_ORIGINS`.

---

## 2. Daftar Endpoint API

| Method | Endpoint | Deskripsi | Auth |
|--------|----------|-----------|------|
| `GET` | `/health` | Cek server hidup | — |
| `POST` | `/api/scan` | Upload foto KTM → dapat NIM + nama | — |
| `POST` | `/api/face/enroll` | Daftarkan wajah baru + check-in | — |
| `POST` | `/api/face/verify` | Verifikasi wajah harian + check-in | — |
| `POST` | `/api/checkout/{nim}` | Check-out mahasiswa dari lab | — |
| `DELETE` | `/api/face/{nim}` | [ADMIN] Reset wajah mahasiswa | `X-Admin-Key` |
| `GET` | `/api/status` | [ADMIN] Lihat peminjaman aktif | `X-Admin-Key` |
| `WS` | `/ws/scan` | WebSocket real-time KTM scanning | — |

---

## 3. Alur Utama (User Flow)

```
┌──────────┐    POST /api/scan     ┌──────────┐
│  Scan    │ ────────────────────► │  Backend │
│  KTM     │    (upload foto)      │  ML      │
└──────────┘                       └────┬─────┘
                                        │
                                        ▼
                              action_required = ?
                                   │         │
                          "face_enroll"   "face_verify"
                                   │         │
                                   ▼         ▼
                         POST /api/face/   POST /api/face/
                            enroll           verify
                                   │         │
                                   ▼         ▼
                              ✅ Check-in berhasil
```

---

## 4. Contoh Kode JavaScript

### 4.1 Scan KTM (Upload Foto)

```javascript
async function scanKTM(imageBlob) {
  const formData = new FormData();
  formData.append("file", imageBlob, "ktm.jpg");

  const res = await fetch("http://localhost:8000/api/scan", {
    method: "POST",
    body: formData,
  });

  const data = await res.json();
  
  // Contoh response sukses:
  // {
  //   "nim_final": "J0403231061",
  //   "nama": "RAIHAN ZALDI",
  //   "db_verified": true,
  //   "db_nama": "Raihan Zaldi",
  //   "action_required": "face_enroll"  ← PENTING
  // }

  return data;
}
```

### 4.2 Ambil Frame Kamera sebagai Base64

```javascript
function captureFrameAsBase64(videoElement) {
  const canvas = document.createElement("canvas");
  canvas.width = videoElement.videoWidth;
  canvas.height = videoElement.videoHeight;
  canvas.getContext("2d").drawImage(videoElement, 0, 0);
  
  // Hapus prefix "data:image/jpeg;base64,"
  return canvas.toDataURL("image/jpeg", 0.8).split(",")[1];
}
```

### 4.3 Enroll Wajah (Pertama Kali)

```javascript
async function enrollFace(nim, nama, base64Image) {
  const res = await fetch("http://localhost:8000/api/face/enroll", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      nim: nim,
      nama: nama,
      image_base64: base64Image,
    }),
  });

  if (res.status === 200) {
    const data = await res.json();
    // { status: "enrolled", nim: "...", checkin: { success: true } }
    alert("✅ Wajah berhasil didaftarkan!");
    return data;
  }
  
  // Error handling
  const err = await res.json();
  if (res.status === 409) alert("Wajah sudah terdaftar.");
  if (res.status === 422) alert("Wajah tidak terdeteksi. Coba lagi.");
  if (res.status === 400) alert("Gambar tidak valid.");
  
  return null;
}
```

### 4.4 Verify Wajah (Absensi Harian)

```javascript
async function verifyFace(nim, base64Image) {
  const res = await fetch("http://localhost:8000/api/face/verify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      nim: nim,
      image_base64: base64Image,
    }),
  });

  if (res.status === 200) {
    const data = await res.json();
    // { status: "verified", similarity: 0.85, checkin: {...} }
    alert(`✅ Selamat datang, ${data.nama}! (${Math.round(data.similarity * 100)}% cocok)`);
    return data;
  }
  
  if (res.status === 403) {
    const err = await res.json();
    alert(`❌ Wajah tidak cocok (${Math.round(err.similarity * 100)}%)`);
  }
  
  return null;
}
```

### 4.5 Checkout

```javascript
async function checkout(nim) {
  const res = await fetch(`http://localhost:8000/api/checkout/${nim}`, {
    method: "POST",
  });
  return await res.json();
  // { success: true, message: "✅ Check-out berhasil..." }
}
```

### 4.6 Admin: Lihat Status Lab

```javascript
async function getLabStatus(adminKey) {
  const res = await fetch("http://localhost:8000/api/status", {
    headers: { "X-Admin-Key": adminKey },
  });
  
  if (res.status === 401) {
    alert("Unauthorized! API Key salah.");
    return null;
  }
  
  return await res.json();
  // { active_count: 3, peminjaman: [...] }
}
```

### 4.7 Admin: Reset Wajah

```javascript
async function resetFace(nim, adminKey) {
  const res = await fetch(`http://localhost:8000/api/face/${nim}`, {
    method: "DELETE",
    headers: { "X-Admin-Key": adminKey },
  });
  return await res.json();
}
```

---

## 5. HTTP Status Codes (Error Handling)

| Status | Kapan Muncul | Apa yang Harus Dilakukan di UI |
|--------|-------------|-------------------------------|
| `200` | Sukses | Tampilkan hasil / konfirmasi ✅ |
| `400` | Gambar rusak / NIM invalid | Tampilkan "Format tidak valid" |
| `401` | Admin key salah / tidak ada | Tampilkan "Akses ditolak" |
| `403` | Wajah TIDAK cocok | Tampilkan "Verifikasi gagal" + similarity % |
| `404` | NIM / wajah tidak ditemukan | Tampilkan "Data tidak ditemukan" |
| `409` | Wajah sudah terdaftar | Redirect ke `/api/face/verify` |
| `413` | File terlalu besar (>10MB) | Tampilkan "Gambar terlalu besar" |
| `422` | Wajah tidak terdeteksi | Tampilkan "Pastikan wajah terlihat jelas" |
| `503` | Model ML belum ready | Tampilkan "Server sedang loading, tunggu..." |

---

## 6. Alur Logika Frontend (Pseudocode)

```
FUNGSI utama():
    1. Buka kamera
    2. User tekan tombol "Scan KTM"
    3. Ambil foto → kirim ke POST /api/scan
    4. Terima response:
       - Jika db_verified = false → tampilkan "KTM tidak dikenali"
       - Jika action_required = "face_enroll":
           a. Tampilkan UI "Pertama kali? Daftarkan wajah Anda"
           b. User tekan "Daftarkan" → ambil frame kamera
           c. Kirim ke POST /api/face/enroll
           d. Jika 200 → "Selamat datang! Wajah terdaftar ✅"
       - Jika action_required = "face_verify":
           a. Tampilkan UI "Verifikasi wajah..."
           b. Otomatis ambil frame kamera
           c. Kirim ke POST /api/face/verify
           d. Jika 200 → "Check-in berhasil ✅"
           e. Jika 403 → "Wajah tidak cocok ❌"
    5. Untuk checkout: tombol terpisah → POST /api/checkout/{nim}
```

---

## 7. Tips Penting

1. **Simpan `nim` dan `nama`** dari response `/api/scan` ke state — akan dipakai di langkah selanjutnya
2. **Kualitas gambar wajah** sangat berpengaruh — pastikan resolusi minimal 320x320px dan pencahayaan cukup
3. **Jangan kirim frame terlalu sering** — 1 frame per detik sudah cukup untuk verifikasi
4. **Admin key default** untuk development: `smartlab-admin-2025` (jangan pakai di production!)
5. **Test manual** bisa di Swagger UI: buka `http://localhost:8000/docs`
