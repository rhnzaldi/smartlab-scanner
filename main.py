"""
Smart-Lab SV IPB — FastAPI Backend
REST API + WebSocket endpoint untuk KTM scanning.
"""

import logging
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ml.pipeline import KTMPipeline
from db.database import init_db
from core.dependencies import set_pipeline

from routers import auth, scan, face, peminjaman, status, labs, jadwal, reports

# ────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("smartlab-api")

ALLOWED_ORIGINS = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000"
).split(",")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Smart-Lab ML Engine...")
    init_db()
    logger.info("✅ Database ready")
    
    pipeline = KTMPipeline(model_path="models/best.pt")
    set_pipeline(pipeline)
    
    if pipeline.is_ready():
        logger.info("✅ Pipeline ready!")
    else:
        logger.warning("⚠️ Pipeline loaded but model NOT FOUND.")
    yield
    logger.info("🛑 Shutting down Smart-Lab ML Engine.")
    set_pipeline(None)

app = FastAPI(
    title="Smart-Lab SV IPB — ML Engine",
    description="Real-time KTM scanning via YOLO + OCR pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
)

@app.get("/health")
async def health_check():
    from core.dependencies import get_pipeline
    pipeline = get_pipeline()
    return {
        "status": "ok",
        "model_loaded": pipeline.is_ready() if pipeline else False,
    }

# Kita include dengan prefix="/api" untuk semua REST API, kecuali scan.router yang mengatur absolute path
app.include_router(auth.router, prefix="/api")
app.include_router(face.router, prefix="/api")
app.include_router(peminjaman.router, prefix="/api")
app.include_router(status.router, prefix="/api")
app.include_router(labs.router, prefix="/api")
app.include_router(jadwal.router, prefix="/api")
app.include_router(reports.router, prefix="/api")

# Include scan router tanpa prefix agar /ws/scan dan /api/scan terdaftar sesuai path absolute-nya
app.include_router(scan.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
