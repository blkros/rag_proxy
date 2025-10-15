from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import os, httpx, time, uuid

RAG = os.getenv("RAG_PROXY_URL", "http://rag-proxy:8080")
OPENAI = os.getenv("OPENAI_URL", "http://172.16.10.168:9993/v1")   # vLLM OpenAI 주소
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "qwen3-30b-a3b-fp8")      # vLLM 모델명
ROUTER_MODEL_ID = os.getenv("ROUTER_MODEL_ID", "qwen3-30b-a3b-fp8-router")

app = FastAPI()

class Msg(BaseModel):
    role: str
    content: str

class ChatReq(BaseModel):
    model: str
    messages: List[Msg]
    stream: Optional[bool] = False

def build_system_with_context(ctx_text: str) -> str:
    return (
        "규칙:\n"
        "- 한국어로 간결하게 답한다.\n"
        "- 아래 컨텍스트 문장들에서만 답을 만든다. 없으면 정확히 `인덱스에 근거 없음`만 출력한다.\n"
        "- 일반 지식/추측 금지, 출처/내부로그 노출 금지.\n\n"
        "[컨텍스트 시작]\n"
        f"{ctx_text}\n"
        "[컨텍스트 끝]\n"
    )

@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": ROUTER_MODEL_ID, "object": "model"}]}

@app.post("/v1/chat/completions")
async def chat(req: ChatReq):
    user_msg = next((m.content for m in reversed(req.messages) if m.role == "user"), "").strip()

    async with httpx.AsyncClient(timeout=None) as client:
        qa = await client.post(f"{RAG}/qa", json={"q": user_msg, "k": 5})
    qa_json = qa.json()

    hits = qa_json.get("hits") or 0
    if hits <= 0:
        content = "인덱스에 근거 없음"
        return {
            "id": f"cmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        }

    items = qa_json.get("items", [])
    ctx_text = "\n\n".join([it.get("text", "") for it in items])[:8000]
    system_prompt = build_system_with_context(ctx_text)

    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "system", "content": system_prompt}] + [m.model_dump() for m in req.messages],
        "stream": False,
        "temperature": 0
    }

    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(f"{OPENAI}/chat/completions", json=payload)
    rj = r.json()
    content = rj.get("choices", [{}])[0].get("message", {}).get("content", "") or "인덱스에 근거 없음"

    return {
        "id": f"cmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
    }