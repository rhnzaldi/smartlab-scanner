import asyncio
from fastapi import APIRouter, Depends, HTTPException
from db.database import check_out, approve_peminjaman, reject_peminjaman, get_active_peminjaman
from core.dependencies import get_current_admin, get_current_student
from core.utils import validate_nim

router = APIRouter(tags=["Peminjaman"])

@router.post("/checkout/{nim}", dependencies=[Depends(get_current_admin)])
async def api_checkout(nim: str):
    nim = validate_nim(nim)
    result = await asyncio.to_thread(check_out, nim)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("message", "Gagal checkout."))
    return result

@router.post("/peminjaman/{pid}/approve", dependencies=[Depends(get_current_admin)])
async def api_approve_peminjaman(pid: int):
    result = await asyncio.to_thread(approve_peminjaman, pid)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result["message"])
    return result

@router.post("/peminjaman/{pid}/reject", dependencies=[Depends(get_current_admin)])
async def api_reject_peminjaman(pid: int):
    result = await asyncio.to_thread(reject_peminjaman, pid)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result["message"])
    return result

@router.get("/me/peminjaman")
async def api_my_peminjaman(current_user: dict = Depends(get_current_student)):
    peminjaman_list = await asyncio.to_thread(get_active_peminjaman)
    my_nim = current_user["username"]
    my_peminjaman = [p for p in peminjaman_list if p["nim"].upper() == my_nim.upper()]
    return {
        "nim": my_nim,
        "name": current_user.get("nama"),
        "peminjaman": my_peminjaman,
        "active_count": len([p for p in my_peminjaman if p["status"] == "aktif"]),
        "pending_count": len([p for p in my_peminjaman if p["status"] == "menunggu"]),
    }
