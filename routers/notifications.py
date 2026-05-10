import asyncio
from fastapi import APIRouter, Depends, HTTPException

from core.dependencies import get_current_admin, get_current_user, get_current_student
from db.database import (
    get_notifications_admin,
    get_notifications_mahasiswa,
    mark_notification_done_for_user,
)

router = APIRouter(tags=["Notifications"])


@router.get("/notifications", dependencies=[Depends(get_current_admin)])
async def api_admin_notifications():
    items = await asyncio.to_thread(get_notifications_admin, 80)
    return {"notifications": items}


@router.get("/notifications/mahasiswa")
async def api_mahasiswa_notifications(current_user: dict = Depends(get_current_student)):
    nim = current_user.get("username") or ""
    items = await asyncio.to_thread(get_notifications_mahasiswa, nim, 80)
    return {"notifications": items}


@router.post("/notifications/{notif_id}/mark-done", dependencies=[Depends(get_current_user)])
async def api_mark_notification_done(notif_id: int, current_user: dict = Depends(get_current_user)):
    role = current_user.get("role") or ""
    username = current_user.get("username") or ""
    result = await asyncio.to_thread(mark_notification_done_for_user, notif_id, role, username)
    if not result.get("success"):
        if "tidak ditemukan" in (result.get("message") or "").lower():
            raise HTTPException(status_code=404, detail=result.get("message"))
        raise HTTPException(status_code=403, detail=result.get("message"))
    return result


@router.get("/public/notifications/pending")
async def api_public_pending_notifications():
    return []
