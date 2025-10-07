# api/smart_router.py
from fastapi import APIRouter, Body, HTTPException
import httpx

router = APIRouter()

@router.post("/ask/smart")
async def ask_smart(payload: dict = Body(...)):
    q = (payload.get("question") or "").strip()
    if not q:
        raise HTTPException(400, "question required")

    async with httpx.AsyncClient() as client:
        r = await client.post("http://localhost:8080/query",
                              json={"question": q, "need_fallback": True})
        return r.json()