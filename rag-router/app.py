from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import os, httpx, time, uuid, re

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

def strip_reasoning(text: str) -> str:
    if not text:
        return text
    # DeepSeek/R1, 일부 Qwen 계열
    text = re.sub(r'(?is)<think>.*?</think>\s*', '', text)
    # Qwen 내부 토큰 스타일
    text = re.sub(r'(?is)<\|assistant_thought\|>.*?(?=<\|assistant_response\|>|\Z)', '', text)
    text = re.sub(r'(?is)<\|assistant_response\|>', '', text)
    # 혹시 남는 “Thought:”류 흔적 간단 정리(선택)
    text = re.sub(r'(?im)^\s*(thought|reasoning)\s*:\s*.*?(?:\n\n|\Z)', '', text)
    return text.strip()

def build_system_with_context(ctx_text: str) -> str:
    return (
        "규칙:\n"
        "- 한국어로 간결하게 답한다.\n"
        "- 아래 컨텍스트 문장들에서만 답을 만든다. 없으면 정확히 `인덱스에 근거 없음`만 출력한다.\n"
        "- 일반 지식/추측 금지, 출처/내부로그 노출 금지.\n"
        "- **내부 추론/작업 메모/체인 오브 소트(예: <think>...</think>)는 절대 출력하지 말고 최종 답변만 출력한다.**\n\n"
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

    # 1) 먼저 /qa 로 컨텍스트 검색
    async with httpx.AsyncClient(timeout=None) as client:
        qa = await client.post(f"{RAG}/qa", json={"q": user_msg, "k": 5})
    qa_json = qa.json()

    hits = qa_json.get("hits") or 0

    # ─────────────────────────────────────────────────────────────
    # ▶▶ [수정 포인트 1: 폴백 추가]
    #    검색 적중이 0이면 rag-proxy의 /v1/chat/completions 를 호출해서
    #    MCP(Confluence) 폴백을 트리거하고, 그 완성 답변을 그대로 반환.
    #    (strip_reasoning 으로 <think> 등 제거)
    # ─────────────────────────────────────────────────────────────
    if hits <= 0:
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                r = await client.post(
                    f"{RAG}/v1/chat/completions",
                    json={
                        # rag-proxy 내부에서 OPENAI_MODEL 로 알아서 호출됨
                        "model": ROUTER_MODEL_ID,
                        "messages": [m.model_dump() for m in req.messages],
                        "stream": False,
                        "temperature": 0
                    }
                )
            rj = r.json()
            raw = rj.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
            content = strip_reasoning(raw) or "인덱스에 근거 없음"
        except Exception:
            # rag-proxy 호출 실패 등 예외 시엔 안전하게 기존 메시지 유지
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
            }],
        }
    # ─────────────────────────────────────────────────────────────

    # 2) 적중이 있으면 기존 방식대로 컨텍스트를 system 메시지에 주입해서 vLLM 호출
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
    raw = rj.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    content = strip_reasoning(raw) or "인덱스에 근거 없음"

    return {
        "id": f"cmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
    }