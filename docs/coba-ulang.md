1.Buat venv
( Python 3.10 / 3.11)
python -m venv venv

aktifin venv
.\venv\Scripts\activate

Cek versi Python
python --version

2.install library
python -m pip install --upgrade pip

install semua dependencies
pip install -r requirements.txt

3,kalau pzybar eror kaya kemarenpip install pyzbar[scripts]
4. coba ini aja kalau pake gpupip uninstall onnxruntime -y
pip install onnxruntime-directmlkalau gagal hapus pip uninstall onnxruntime-directml -y
pip install onnxruntime

5.seederpython seed_mahasiswa.py

6.jalanin backend
.\venv\Scripts\activate

server FastAPI
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

7. kalau mau jalan ulang
cd ScanKtm

.\venv\Scripts\activate

uvicorn main:app --host 0.0.0.0 --port 8000 --reload

python test_webcam.py