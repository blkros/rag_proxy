from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import os, httpx, time, uuid, re
from html import unescape  # ← (추가) HTML 엔티티 정리용

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
        "- 아래 컨텍스트에 있는 정보만 사용한다.\n"
        "- 질문과 완벽히 일치하지 않아도, 컨텍스트와 관련된 사실이 있으면 그 범위 안에서 요약/발췌해 답한다.\n"
        "- 컨텍스트에 없는 내용은 추측하지 않는다.\n"
        "- **내부 추론/체인오브소트(예: <think>...</think>)는 절대 출력하지 말고 최종 답변만 출력한다.**\n"
        "- 컨텍스트가 완전히 비었거나 전혀 관련이 없을 때만 정확히 `인덱스에 근거 없음`을 출력한다.\n"
        "- 부분 일치 시에는 '컨텍스트에서 확인된 항목:'으로 시작해 제공된 항목만 정리한다.\n\n"
        "[컨텍스트 시작]\n"
        f"{ctx_text}\n"
        "[컨텍스트 끝]\n"
    )

def extract_texts(items: List[dict]) -> List[str]:
    """다양한 키에서 본문을 뽑아 HTML 엔티티도 정리"""
    texts = []
    for it in items or []:
        for key in ("text", "content", "chunk", "snippet", "body", "page_text"):
            val = it.get(key)
            if isinstance(val, str) and val.strip():
                texts.append(unescape(val.strip()))  # ← (변경) unescape
                break
        else:
            payload = it.get("payload") or it.get("data") or {}
            if isinstance(payload, dict):
                for key in ("text", "content", "body"):
                    val = payload.get(key)
                    if isinstance(val, str) and val.strip():
                        texts.append(unescape(val.strip()))  # ← (변경) unescape
                        break
    return texts

def is_good_context_for_qa(ctx: str) -> bool:
    """QA 경로에서만 쓰는 느슨한 품질 체크 (짧으면 /query 폴백)"""
    if not ctx or not ctx.strip():
        return False
    # 너무 짧은 제목/스니펫은 제외 (기존보다 완화)
    if len(ctx) < 180:
        return False
    if ctx.count("\n") < 2:
        return False
    return True

@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": ROUTER_MODEL_ID, "object": "model"}]}

@app.post("/v1/chat/completions")
async def chat(req: ChatReq):
    user_msg = next((m.content for m in reversed(req.messages) if m.role == "user"), "").strip()

    # 1) /qa 먼저
    async with httpx.AsyncClient(timeout=None) as client:
        qa = await client.post(f"{RAG}/qa", json={"q": user_msg, "k": 5})
    qa_json = qa.json()

    hits = qa_json.get("hits") or 0
    qa_items = qa_json.get("items", []) or qa_json.get("contexts", [])
    qa_ctx = "\n\n".join(extract_texts(qa_items))[:8000]
    qa_ctx = unescape(qa_ctx)  # ← (추가) 혹시 몰라 한 번 더 정리

    # 2-A) /qa 결과가 “있고+적당히 길다”면 그걸로 LLM
    if hits > 0 and is_good_context_for_qa(qa_ctx):
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
        content = strip_reasoning(raw).strip()
        if content == "인덱스에 근거 없음" and ctx_text.strip():
            # 너무 장문이면 앞부분만
            safe_ctx = ctx_text.strip().replace("\u00a0", " ")[:600]
            content = "컨텍스트에서 확인된 항목:\n" + safe_ctx

        return {
            "id": f"cmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content or "인덱스에 근거 없음"}, "finish_reason": "stop"}],
        }

    # 2-B) /qa가 부실(짧음/없음) → 무조건 /query 폴백 (MCP-Confluence 포함)
    async with httpx.AsyncClient(timeout=None) as client:
        qres = await client.post(f"{RAG}/query", json={"q": user_msg, "k": 5})
    qj = qres.json()

    ctx_list = []
    if isinstance(qj.get("context_texts"), list):
        ctx_list.extend([unescape(t) for t in qj["context_texts"] if isinstance(t, str) and t.strip()])
    if isinstance(qj.get("contexts"), list):
        ctx_list.extend(extract_texts(qj["contexts"]))
    if isinstance(qj.get("items"), list):
        ctx_list.extend(extract_texts(qj["items"]))

    ctx_text = "\n\n---\n\n".join([t for t in ctx_list if t])[:8000]
    ctx_text = unescape(ctx_text)  # ← (추가) 최종 정리

    # 폴백 경로에서는 “길이 체크 없이”, 1자라도 있으면 LLM 호출
    if not ctx_text.strip():
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