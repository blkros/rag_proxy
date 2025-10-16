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
    orig_user_msg = next((m.content for m in reversed(req.messages) if m.role == "user"), "").strip()
    user_msg = normalize_query(orig_user_msg)

    ctx_text = ""  # [FIX-1] 어떤 코드 경로에서도 참조 가능하도록 기본값으로 초기화

    # 1) 내부 RAG(/qa) 빠른 조회
    async with httpx.AsyncClient(timeout=None) as client:
        qa = await client.post(f"{RAG}/qa", json={"q": user_msg, "k": 5})
    qa_json = qa.json()

    hits = qa_json.get("hits") or 0
    items = qa_json.get("items", [])

    # 정규화 히트 없으면 원문으로 재시도 (선택)
    if hits <= 0 and user_msg != orig_user_msg:
        async with httpx.AsyncClient(timeout=None) as client:
            qa2 = await client.post(f"{RAG}/qa", json={"q": orig_user_msg, "k": 5})
        qa2_json = qa2.json()
        if (qa2_json.get("hits") or 0) > 0:
            qa_json = qa2_json
            hits = qa_json["hits"]
            items = qa_json.get("items", [])

    # 2-A) FAISS 등에서 뭔가 나오면: 그 컨텍스트로 LLM 호출
    if hits > 0:
        ctx_text = "\n\n".join([it.get("text", "") for it in items])[:8000]
        system_prompt = build_system_with_context(ctx_text)

        payload = {
            "model": OPENAI_MODEL,
            "messages": [{"role": "system", "content": system_prompt}] + [m.model_dump() for m in req.messages],
            "stream": False,
            "temperature": 0,
        }
        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(f"{OPENAI}/chat/completions", json=payload)

        rj = r.json()
        raw = rj.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        content = strip_reasoning(raw).strip() or "인덱스에 근거 없음"

        # [FIX-2] ctx_text는 이 블록 안에서만 안전하게 참조
        if content == "인덱스에 근거 없음" and ctx_text.strip():
            content = "컨텍스트에서 확인된 항목:\n" + sanitize(ctx_text)[:600]

        return {
            "id": f"cmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        }

    # 2-B) FAISS가 0건이면: /query 폴백 → MCP-Confluence까지 돌린 컨텍스트 사용
    async with httpx.AsyncClient(timeout=None) as client:
        qres = await client.post(f"{RAG}/query", json={"q": user_msg, "k": 5})
    qj = qres.json()

    ctx_list = (
        qj.get("context_texts")
        or [c.get("text", "") for c in (qj.get("contexts") or [])]
        or [it.get("text", "") for it in (qj.get("items") or [])]
    )
    ctx_text = "\n\n---\n\n".join([t for t in ctx_list if t])[:8000]  # [중요] 여기서도 ctx_text를 **설정**

    if not ctx_text:
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
        "temperature": 0,
    }
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(f"{OPENAI}/chat/completions", json=payload)

    rj = r.json()
    raw = rj.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    content = strip_reasoning(raw).strip() or "인덱스에 근거 없음"

    # [FIX-2] 마찬가지로 이 블록 안에서만 ctx_text 사용
    if content == "인덱스에 근거 없음" and ctx_text.strip():
        content = "컨텍스트에서 확인된 항목:\n" + sanitize(ctx_text)[:600]

    return {
        "id": f"cmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
    }