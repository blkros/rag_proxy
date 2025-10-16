from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import os, httpx, time, uuid, re

RAG = os.getenv("RAG_PROXY_URL", "http://rag-proxy:8080")  # <- 라우터 컨테이너 내부에선 서비스명 OK
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
        "- **내부 추론/작업 메모/체인오브소트(예: <think>...</think>)는 절대 출력하지 말고 최종 답변만 출력한다.**\n\n"
        "[컨텍스트 시작]\n"
        f"{ctx_text}\n"
        "[컨텍스트 끝]\n"
    )

def extract_texts(items: List[dict]) -> List[str]:
    """/qa 또는 /query의 item 배열에서 본문 텍스트를 폭넓게 추출"""
    texts = []
    for it in items or []:
        for key in ("text", "content", "chunk", "snippet", "body", "page_text"):
            val = it.get(key)
            if isinstance(val, str) and val.strip():
                texts.append(val.strip())
                break
        else:
            payload = it.get("payload") or it.get("data") or {}
            if isinstance(payload, dict):
                for key in ("text", "content", "body"):
                    val = payload.get(key)
                    if isinstance(val, str) and val.strip():
                        texts.append(val.strip())
                        break
    return texts

def is_good_context(ctx: str) -> bool:
    """컨텍스트가 제목/짧은 스니펫 수준인지 판별해서 부실하면 False 반환"""
    if not ctx or not ctx.strip():
        return False
    # 길이/줄수/문장부호 기준의 간단 휴리스틱 (필요시 조정)
    if len(ctx) < 250:
        return False
    if ctx.count("\n") < 5:
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

    # 2-A) /qa 결과가 “있고 + 충분히 길다”면 그걸로 바로 LLM
    if hits > 0 and is_good_context(qa_ctx):
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

    # 2-B) /qa가 부실(짧음/없음) → 반드시 /query 폴백 (여기서 MCP-Confluence가 도는 구간)
    async with httpx.AsyncClient(timeout=None) as client:
        qres = await client.post(f"{RAG}/query", json={"q": user_msg, "k": 5})
    qj = qres.json()

    ctx_list = []
    if isinstance(qj.get("context_texts"), list):
        ctx_list.extend([t for t in qj["context_texts"] if isinstance(t, str) and t.strip()])
    if isinstance(qj.get("contexts"), list):
        ctx_list.extend(extract_texts(qj["contexts"]))
    if isinstance(qj.get("items"), list):
        ctx_list.extend(extract_texts(qj["items"]))

    ctx_text = "\n\n---\n\n".join(ctx_list)[:8000]

    if not is_good_context(ctx_text):
        # MCP까지 돌렸는데도 내용이 없다면 진짜로 없음
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