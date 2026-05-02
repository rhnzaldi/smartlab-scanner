import asyncio
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from db.database import get_jadwal, create_jadwal, update_jadwal, delete_jadwal, archive_all_jadwal
from core.dependencies import get_current_admin

router = APIRouter(tags=["Jadwal"])

class SchedulePayload(BaseModel):
    mata_kuliah: str
    kelas: Optional[str] = None
    prodi: Optional[str] = None
    lab: Optional[str] = None
    gedung: Optional[str] = None
    hari: Optional[str] = None
    jam_mulai: Optional[str] = None
    jam_selesai: Optional[str] = None
    tipe_semester: Optional[str] = None
    tahun_ajaran: Optional[str] = None
    status: Optional[str] = "aktif"

@router.get("/jadwal", dependencies=[Depends(get_current_admin)])
async def api_get_jadwal(archived: bool = False):
    return await asyncio.to_thread(get_jadwal, archived)

@router.get("/public/jadwal")
async def api_public_jadwal(archived: bool = False):
    return await asyncio.to_thread(get_jadwal, archived)

@router.post("/jadwal", dependencies=[Depends(get_current_admin)])
async def api_create_jadwal(payload: SchedulePayload):
    return await asyncio.to_thread(
        create_jadwal,
        payload.mata_kuliah,
        payload.kelas,
        payload.prodi,
        payload.lab,
        payload.gedung,
        payload.hari,
        payload.jam_mulai,
        payload.jam_selesai,
        payload.tipe_semester,
        payload.tahun_ajaran,
        payload.status,
    )

@router.put("/jadwal/{jadwal_id}", dependencies=[Depends(get_current_admin)])
async def api_update_jadwal(jadwal_id: int, payload: SchedulePayload):
    updated = await asyncio.to_thread(
        update_jadwal,
        jadwal_id,
        payload.mata_kuliah,
        payload.kelas,
        payload.prodi,
        payload.lab,
        payload.gedung,
        payload.hari,
        payload.jam_mulai,
        payload.jam_selesai,
        payload.tipe_semester,
        payload.tahun_ajaran,
        payload.status,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Jadwal tidak ditemukan")
    return updated

@router.delete("/jadwal/{jadwal_id}", dependencies=[Depends(get_current_admin)])
async def api_delete_jadwal(jadwal_id: int):
    success = await asyncio.to_thread(delete_jadwal, jadwal_id)
    if not success:
        raise HTTPException(status_code=404, detail="Jadwal tidak ditemukan")
    return {"success": True}

@router.post("/jadwal/archive", dependencies=[Depends(get_current_admin)])
async def api_archive_jadwal():
    count = await asyncio.to_thread(archive_all_jadwal)
    return {"success": True, "archived_count": count}
