import asyncio
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from db.database import get_labs, create_lab, update_lab, delete_lab
from core.dependencies import get_current_admin

router = APIRouter(tags=["Labs"])

class LabPayload(BaseModel):
    name: str
    location: Optional[str] = None
    capacity: Optional[int] = None
    op_start: Optional[str] = None
    op_end: Optional[str] = None
    use_start: Optional[str] = None
    use_end: Optional[str] = None
    equipment: Optional[list] = None
    status_override: Optional[str] = None

@router.get("/labs", dependencies=[Depends(get_current_admin)])
async def api_get_labs():
    return await asyncio.to_thread(get_labs)

@router.post("/labs", dependencies=[Depends(get_current_admin)])
async def api_create_lab(payload: LabPayload):
    return await asyncio.to_thread(
        create_lab,
        payload.name,
        payload.location,
        payload.capacity,
        payload.op_start,
        payload.op_end,
        payload.use_start,
        payload.use_end,
        payload.equipment or [],
        payload.status_override,
    )

@router.put("/labs/{lab_id}", dependencies=[Depends(get_current_admin)])
async def api_update_lab(lab_id: int, payload: LabPayload):
    updated = await asyncio.to_thread(
        update_lab,
        lab_id,
        payload.name,
        payload.location,
        payload.capacity,
        payload.op_start,
        payload.op_end,
        payload.use_start,
        payload.use_end,
        payload.equipment or [],
        payload.status_override,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Lab tidak ditemukan")
    return updated

@router.delete("/labs/{lab_id}", dependencies=[Depends(get_current_admin)])
async def api_delete_lab(lab_id: int):
    success = await asyncio.to_thread(delete_lab, lab_id)
    if not success:
        raise HTTPException(status_code=404, detail="Lab tidak ditemukan")
    return {"success": True}
