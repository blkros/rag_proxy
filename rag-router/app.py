from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import os, httpx, time, uuid, re

RAG = os.getenv("RAG_PROXY_URL", "http://rag-proxy:8080")
OPENAI = os.getenv("OPENAI_URL", "http://172.16.10.168:9993/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "qwen3-30b-a3b-fp8")
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
    text = re.sub(r'(?is)<think>.*?</think>\s*', '', text)
    text = re.sub(r'(?is)<\|assistant_thought\|>.*?(?=<\|assistant_response\|>|\Z)', '', text)
    text = re.sub(r'(?is)<\|assistant_response\|>', '', text)
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

def extract_texts(items: List[dict]) -> List[str]:
    """ /qa 혹은 /query 아이템에서 가능한 모든 키를 시도해 텍스트를 뽑는다. """
    texts = []
    for it in items or []:
        # 우선순위: text -> content -> chunk -> snippet -> body -> page_text
        for key in ("text", "content", "chunk", "snippet", "body", "page_text"):
            val = it.get(key)
            if isinstance(val, str) and val.strip():
                texts.append(val.strip())
                break
        else:
            # 중첩 구조 대비 (예: {"payload":{"text":"..."}})
            payload = it.get("payload") or it.get("data") or {}
            if isinstance(payload, dict):
                for key in ("text", "content", "body"):
                    val = payload.get(key)
                    if isinstance(val, str) and val.strip():
                        texts.append(val.strip())
                        break
    return texts

@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": ROUTER_MODEL_ID, "object": "model"}]}

@app.post("/v1/chat/completions")
async def chat(req: ChatReq):
    user_msg = next((m.content for m in reversed(req.messages) if m.role == "user"), "").strip()

    # 1) 내부 RAG(/qa)
    async with httpx.AsyncClient(timeout=None) as client:
        qa = await client.post(f"{RAG}/qa", json={"q": user_msg, "k": 5})
    qa_json = qa.json()

    hits = qa_json.get("hits") or 0
    qa_items = qa_json.get("items", []) or qa_json.get("contexts", [])  # 일부 구현은 contexts로 내려줄 수 있음
    qa_texts = extract_texts(qa_items)
    qa_ctx = "\n\n".join(qa_texts)[:8000]

    # 2-A) /qa hits>0 이라도 컨텍스트가 비어 있으면 → /query 폴백 진입
    if hits > 0 and qa_ctx.strip():
        system_prompt = build_system_with_context(qa_ctx)
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
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        }

    # 2-B) /query 폴백 (여기서 MCP-Confluence가 rag-proxy 내부에서 동작)
    async with httpx.AsyncClient(timeout=None) as client:
        qres = await client.post(f"{RAG}/query", json={"q": user_msg, "k": 5})
    qj = qres.json()

    # /query는 구현에 따라 다양한 필드를 가질 수 있음 → 최대한 폭넓게 집계
    ctx_list = []
    if isinstance(qj.get("context_texts"), list):
        ctx_list.extend([t for t in qj["context_texts"] if isinstance(t, str) and t.strip()])
    if isinstance(qj.get("contexts"), list):
        ctx_list.extend(extract_texts(qj["contexts"]))
    if isinstance(qj.get("items"), list):
        ctx_list.extend(extract_texts(qj["items"]))

    ctx_text = "\n\n---\n\n".join(ctx_list)[:8000]

    if not ctx_text.strip():
        # MCP까지 돌았는데도 컨텍스트가 없다면 진짜로 없음
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
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
    }