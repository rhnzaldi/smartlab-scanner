import re
from fastapi import HTTPException
from db.database import NIM_PATTERN

_NIM_RE = re.compile(NIM_PATTERN)

def validate_nim(nim: str) -> str:
    nim = nim.strip().upper()
    if not _NIM_RE.match(nim):
        raise HTTPException(
            status_code=400,
            detail=f"Format NIM tidak valid: '{nim}'."
        )
    return nim
