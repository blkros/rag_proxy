# api/smart_router.py
from fastapi import APIRouter, Body, HTTPException
import httpx, time, os

router = APIRouter()
RAG_URL = os.getenv("RAG_URL", "http://localhost:8080")

@router.post("/ask/smart")
async def ask_smart(payload: dict = Body(...)):
    q = (payload.get("question") or payload.get("q") or "").strip()
    if not q:
        raise HTTPException(400, "question required")

    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(f"{RAG_URL}/query", json={"question": q})
    data = r.json()

    # ★ 핵심 패치: direct_answer가 있으면 그걸 바로 assistant 메시지로 리턴
    if isinstance(data, dict) and data.get("direct_answer"):
        return {
            "object": "chat.completion",
            "model": "qwen3-30b-a3b-fp8-router",
            "created": int(time.time()),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": data["direct_answer"]},
                "finish_reason": "stop",
            }],
            "notes": data.get("notes", {}),
        }

    # direct_answer가 없으면 원래처럼 그대로 반환(또는 이후 RAG/LLM 호출 분기 로직)
    return data
