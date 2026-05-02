import asyncio
from typing import Optional
from fastapi import APIRouter, Depends
from db.database import get_peminjaman_history
from core.dependencies import get_current_admin

router = APIRouter(tags=["Reports"])

@router.get("/reports/peminjaman", dependencies=[Depends(get_current_admin)])
async def api_get_peminjaman_history(
    year: Optional[int] = None,
    month: Optional[int] = None,
    prodi: Optional[str] = None,
):
    return await asyncio.to_thread(get_peminjaman_history, year, month, prodi)
