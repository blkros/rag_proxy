import os, time, uuid, json, re
from typing import List, Optional, Dict, Any
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ===== 환경변수 =====
RAG_PROXY_URL   = os.getenv("RAG_PROXY_URL", "http://rag-proxy:8080")
# Ollama(기본) — Open WebUI 로그에 ollama 체크가 떠서 기본을 Ollama로 둡니다.
OLLAMA_URL      = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "qwen3:30b")
# 라우터 상에서 표시될 모델 ID (Open WebUI 모델 목록용)
ROUTER_MODEL_ID = os.getenv("ROUTER_MODEL_ID", "qwen3-30b-a3b-fp8-router")

# 컨텍스트 최대 바이트(너무 길면 LLM 호출 실패 방지)
MAX_CTX_CHARS   = int(os.getenv("MAX_CTX_CHARS", "8000"))
RETRIEVE_K      = int(os.getenv("RETRIEVE_K", "5"))

app = FastAPI(title="RAG Router", version="1.0.0")

# ====== Pydantic ======
class ChatMsg(BaseModel):
    role: str
    content: str

class ChatCompletionReq(BaseModel):
    model: str
    messages: List[ChatMsg]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.0

# ====== helpers ======
def _last_user_text(messages: List[Dict[str, Any]]) -> str:
    for m in reversed(messages):
        if (m.get("role") or "").lower() == "user" and m.get("content"):
            return str(m["content"]).strip()
    return ""

async def _rag_query(question: str) -> Dict[str, Any]:
    """
    rag-proxy가 이미 /qa에서 내부 폴백(MCP)까지 수행하도록 구성되어 있으므로
    여기서는 /qa에 {'q': ..., 'k': ...}만 던집니다.
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"{RAG_PROXY_URL}/qa", json={"q": question, "k": RETRIEVE_K})
        r.raise_for_status()
        return r.json()

async def _ollama_chat(messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
    """
    Ollama /api/chat 호출. stream=False로 단답 처리(웹UI는 비스트림도 수용).
    """
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature}
            }
        )
        r.raise_for_status()
        data = r.json()
    return (data.get("message") or {}).get("content", "") or ""

# ====== OpenAI 호환 엔드포인트 ======
@app.get("/v1/models")
async def v1_models():
    return {
        "object": "list",
        "data": [{
            "id": ROUTER_MODEL_ID,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "rag-router"
        }]
    }

@app.post("/v1/chat/completions")
async def v1_chat_completions(req: ChatCompletionReq):
    # 1) 사용자 메시지 추출
    user_text = _last_user_text([m.model_dump() for m in req.messages])
    if not user_text:
        raise HTTPException(400, "No user message provided.")

    # 2) rag-proxy에서 컨텍스트 확보(벡터 + 필요시 Confluence MCP 폴백)
    try:
        qr = await _rag_query(user_text)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"rag-proxy unreachable: {e}")

    hits = int(qr.get("hits") or 0)
    if hits <= 0:
        content = "인덱스에 근거 없음"
        return {
            "id": f"cmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }]
        }

    # 3) 컨텍스트 구성
    items = qr.get("items") or []
    context = "\n\n".join([str(it.get("text","")) for it in items])[:MAX_CTX_CHARS]

    # 4) LLM 호출 전 system 규칙 삽입
    system_prompt = (
        "규칙:\n"
        "- 한국어로 간결하고 정확하게 답한다.\n"
        "- 아래 컨텍스트에서 근거한 사실만 사용한다.\n"
        "- 컨텍스트로 답할 수 없으면 정확히 `인덱스에 근거 없음`만 출력한다.\n\n"
        "[컨텍스트 시작]\n" + context + "\n[컨텍스트 끝]\n"
    )

    llm_messages = [{"role": "system", "content": system_prompt}] + [
        {"role": m.role, "content": m.content} for m in req.messages
    ]

    # 5) Ollama로 생성
    try:
        answer = await _ollama_chat(llm_messages, temperature=float(req.temperature or 0.0))
        if not answer.strip():
            answer = "인덱스에 근거 없음"
    except httpx.HTTPError as e:
        # LLM 장애 시라도 빈손으로 돌려보내지 말고 최소 메시지 반환
        answer = f"(LLM 오류) 인덱스에 근거 없음"

    return {
        "id": f"cmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop"
        }]
    }
