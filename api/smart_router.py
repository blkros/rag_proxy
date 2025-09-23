# rag-proxy/api/smart_router.py
from __future__ import annotations

from fastapi import APIRouter, Body
from src.fallback_rag import answer_with_fallback

router = APIRouter()


@router.post("/ask/smart")
async def ask_smart(payload: dict = Body(...)):
    q = (payload.get("question") or "").strip()
    space = payload.get("space")
    if not q:
        return {"error": "question required"}
    return answer_with_fallback(q, space=space)