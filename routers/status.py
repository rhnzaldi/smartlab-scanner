import asyncio
from fastapi import APIRouter, Depends
from db.database import get_active_peminjaman, get_current_lab
from core.dependencies import get_current_admin

router = APIRouter(tags=["Status"])

@router.get("/status", dependencies=[Depends(get_current_admin)])
async def api_status():
    peminjaman_list = await asyncio.to_thread(get_active_peminjaman)
    active = [p for p in peminjaman_list if p["status"] == "aktif"]
    pending = [p for p in peminjaman_list if p["status"] == "menunggu"]
    return {
        "active_count": len(active),
        "pending_count": len(pending),
        "peminjaman": active,
        "peminjaman_pending": pending,
    }

@router.get("/public/status")
async def api_public_status():
    peminjaman_list = await asyncio.to_thread(get_active_peminjaman)
    active = [p for p in peminjaman_list if p["status"] == "aktif"]
    pending = [p for p in peminjaman_list if p["status"] == "menunggu"]
    return {
        "active_count": len(active),
        "pending_count": len(pending),
        "peminjaman": active,
        "peminjaman_pending": pending
    }

@router.get("/public/current_lab")
async def api_public_current_lab():
    lab = await asyncio.to_thread(get_current_lab)
    return {"lab": lab}
