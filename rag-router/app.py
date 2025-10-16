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

    # 1) 내부 RAG(/qa)로 빠른 조회
    async with httpx.AsyncClient(timeout=None) as client:
        qa = await client.post(f"{RAG}/qa", json={"q": user_msg, "k": 5})
    qa_json = qa.json()

    hits = qa_json.get("hits") or 0
    items = qa_json.get("items", [])

    # 2-A) FAISS에서라도 뭔가 나오면: 그 컨텍스트로 LLM 직접 호출
    if hits > 0:
        ctx_text = "\n\n".join([it.get("text", "") for it in items])[:8000]
        system_prompt = build_system_with_context(ctx_text)

        payload = {
            "model": OPENAI_MODEL,  # vLLM 직통
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
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        }

    # 2-B) FAISS가 0건이면: rag-proxy의 /query 로 폴백 컨텍스트 수집 → LLM 직접 요약
    #     (여기서 MCP-Confluence가 rag-proxy 내부에서 돌고, 그 결과 텍스트만 받아서 우리가 LLM에 붙임)
    async with httpx.AsyncClient(timeout=None) as client:
        qres = await client.post(f"{RAG}/query", json={"q": user_msg, "k": 5})
    qj = qres.json()

    # /query 응답에서 컨텍스트 텍스트 뽑기: 구현에 따라 'context_texts' 또는 'contexts'/'items' 중 하나가 있음
    ctx_list = (
        qj.get("context_texts")
        or [c.get("text", "") for c in (qj.get("contexts") or [])]
        or [it.get("text", "") for it in (qj.get("items") or [])]
    )
    ctx_text = "\n\n---\n\n".join([t for t in ctx_list if t])[:8000]

    if not ctx_text:
        # MCP까지 돌았는데도 컨텍스트가 없다면 정말로 정보 없음
        content = "인덱스에 근거 없음"
        return {
            "id": f"cmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        }

    system_prompt = build_system_with_context(ctx_text)
    payload = {
        "model": OPENAI_MODEL,  # vLLM 직통
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
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
    }