from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import os, httpx, time, uuid, re
from html import unescape
from datetime import datetime
from zoneinfo import ZoneInfo

RAG = os.getenv("RAG_PROXY_URL", "http://rag-proxy:8080")
OPENAI = os.getenv("OPENAI_URL", "http://172.16.10.168:9993/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "qwen3-30b-a3b-fp8")
ROUTER_MODEL_ID = os.getenv("ROUTER_MODEL_ID", "qwen3-30b-a3b-fp8-router")
TZ = os.getenv("ROUTER_TZ", "Asia/Seoul")

ROUTER_MAX_TOKENS = int(os.getenv("ROUTER_MAX_TOKENS", "2048"))
ANSWER_MODE = os.getenv("ROUTER_ANSWER_MODE", "bulleted")     # bulleted | sections
BULLETS_MAX = int(os.getenv("ROUTER_BULLETS_MAX", "15"))
MAX_CTX_CHARS = int(os.getenv("MAX_CTX_CHARS", "8000"))

# [추가] 제목 접두어 완전 비활성(기본 ""), 필요시 환경변수로 켜도 됨
HEADING = os.getenv("ROUTER_HEADING", "")

# [추가] 출처 표시 on/off 및 최대 개수
ROUTER_SHOW_SOURCES = (os.getenv("ROUTER_SHOW_SOURCES", "1").lower() not in ("0","false","no"))
ROUTER_SOURCES_MAX  = int(os.getenv("ROUTER_SOURCES_MAX", "5"))

app = FastAPI()

class Msg(BaseModel):
    role: str
    content: str

class ChatReq(BaseModel):
    model: str
    messages: List[Msg]
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None

def strip_reasoning(text: str) -> str:
    if not text: return text
    text = re.sub(r'(?is)<think>.*?</think>\s*', '', text)
    text = re.sub(r'(?is)<\|assistant_thought\|>.*?(?=<\|assistant_response\|>|\Z)', '', text)
    text = re.sub(r'(?is)<\|assistant_response\|>', '', text)
    text = re.sub(r'(?im)^\s*(thought|reasoning)\s*:\s*.*?(?:\n\n|\Z)', '', text)
    return text.strip()

# --- utils ----------------------------------------------------

def normalize_query(q: str) -> str:
    if not q: return ""
    s = q.strip()
    s = re.sub(r'(?i)\bstep\b', 'SFTP', s)
    s = re.sub(r'(?i)\bstfp\b|\bsfttp\b|\bsfpt\b|\bsftp\b', 'SFTP', s)
    s = s.replace("스텝", "SFTP")
    return s

def generate_query_variants(q: str, limit: int = 6) -> List[str]:
    s = normalize_query(q)
    cand: List[str] = []
    def add(x: str):
        x = re.sub(r'\s+', ' ', x).strip()
        if x and x not in cand: cand.append(x)
    add(s)
    add(re.sub(r'\s+', '', s))
    add(re.sub(r'([가-힣])([A-Za-z0-9])', r'\1 \2', s))
    add(re.sub(r'([A-Za-z0-9])([가-힣])', r'\1 \2', s))
    pairs = [("개발 서버","개발서버"), ("테스트 서버","테스트서버"), ("운영 서버","운영서버"),
             ("계정 정보","계정정보"), ("접속 정보","접속정보"), ("IP 주소","IP주소")]
    for a,b in pairs:
        add(s.replace(a,b)); add(s.replace(b,a))
    return cand[:limit]

def sanitize(text: str) -> str:
    if not text: return ""
    t = unescape(text.replace("&nbsp;", " "))
    t = re.sub(r'(?i)(password|passwd|pwd|패스워드|비밀번호)\s*[:=]\s*\S+', r'\1: ******', t)
    t = re.sub(r'(?i)(token|secret|key|키)\s*[:=]\s*[A-Za-z0-9\-_]{6,}', r'\1: <redacted>', t)
    t = re.sub(r'(?i)(account|user(?:name)?|userid|계정|아이디)\s*[:=]\s*\S+', r'\1: <redacted>', t)
    t = re.sub(r'\b(\d{1,3}\.\d{1,3}\.\d{1,3})\.\d{1,3}\b', r'\1.xxx', t)
    return t

# [변경] 제목 강제 규칙 제거. 불릿/섹션만 지시(접두어 원하면 HEADING으로만 힌트)
def build_system_with_context(ctx_text: str) -> str:
    style = (
        f"- 최대 {BULLETS_MAX}개 불릿으로 **구체적**으로 서술한다.\n"
        if ANSWER_MODE == "bulleted"
        else "- '개요 / 세부 / 추가 참고' **세 섹션**으로 문단형으로 서술한다.\n"
    )
    heading_hint = (f"- 가능하면 '{HEADING}' 아래로 정리한다.\n" if HEADING else "")
    return (
        "역할: 주어진 컨텍스트를 근거로 **상세하고 실무 친화적인** 한국어 답변을 작성한다.\n"
        "원칙:\n"
        "- 컨텍스트에 있는 정보만 사용하고 추측/환각 금지.\n"
        "- 수치·정책·용어는 가능하면 그대로 인용하되 과도한 반복은 피한다.\n"
        "- 내부 추론(<think> 등) 출력 금지, 최종 답만 출력.\n"
        + heading_hint + style +
        "- 컨텍스트가 완전히 비었거나 무관하면 정확히 `인덱스에 근거 없음`만 출력.\n"
        "- 민감정보(비밀번호/토큰/IP 마지막 옥텟)는 마스킹한다.\n"
        "[컨텍스트 시작]\n"
        f"{ctx_text}\n"
        "[컨텍스트 끝]\n"
    )

def extract_texts(items: List[dict]) -> List[str]:
    texts = []
    for it in items or []:
        for key in ("text","content","chunk","snippet","body","page_text"):
            val = it.get(key)
            if isinstance(val,str) and val.strip():
                texts.append(unescape(val.strip())); break
        else:
            payload = it.get("payload") or it.get("data") or {}
            if isinstance(payload,dict):
                for key in ("text","content","body"):
                    val = payload.get(key)
                    if isinstance(val,str) and val.strip():
                        texts.append(unescape(val.strip())); break
    return texts

# [추가] 아이템/응답에서 URL(출처) 추출
def _collect_urls_from_items(items: List[dict]) -> List[str]:
    urls = []
    def add(u: Optional[str]):
        if not u: return
        u = str(u).strip()
        if u and u not in urls: urls.append(u)
    for it in items or []:
        add(it.get("url") or it.get("source_url") or it.get("link"))
        payload = it.get("payload") or it.get("data") or {}
        if isinstance(payload, dict):
            add(payload.get("url") or payload.get("source_url") or payload.get("link"))
        # rag-proxy 일부 스키마: it["metadata"]["url"]
        meta = it.get("metadata") or {}
        if isinstance(meta, dict):
            add(meta.get("url"))
    # 최대 개수 제한
    return urls[:ROUTER_SOURCES_MAX]

def is_good_context_for_qa(ctx: str) -> bool:
    if not ctx or not ctx.strip(): return False
    if len(ctx) < 180: return False
    if ctx.count("\n") < 2: return False
    return True

@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": ROUTER_MODEL_ID, "object": "model"}]}

@app.post("/v1/chat/completions")
async def chat(req: ChatReq):
    orig_user_msg = next((m.content for m in reversed(req.messages) if m.role == "user"), "").strip()
    variants = generate_query_variants(orig_user_msg)

    ctx_text = ""
    qa_json = None
    qa_items = []
    qa_urls: List[str] = []     # [추가] QA 경로 출처

    timeout = httpx.Timeout(30.0, connect=5.0, read=20.0, write=20.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for v in variants:
            try:
                qa = await client.post(f"{RAG}/qa", json={"q": v, "k": 5})
                j = qa.json()
            except (httpx.RequestError, ValueError) as e:
                print(f"[router] /qa error for '{v}': {e}")
                continue
            if (j.get("hits") or 0) > 0:
                qa_json = j
                qa_items = j.get("items", [])
                qa_urls = _collect_urls_from_items(qa_items)   # [추가]
                break

    # 2-A) QA 성공
    if qa_json:
        ctx_text = "\n\n".join(extract_texts(qa_items))[:MAX_CTX_CHARS]
        if not is_good_context_for_qa(ctx_text):
            qa_json = None

    if qa_json:
        ctx_for_prompt = sanitize(ctx_text)
        system_prompt = build_system_with_context(ctx_for_prompt)
        max_tokens = req.max_tokens or ROUTER_MAX_TOKENS
        payload = {
            "model": OPENAI_MODEL,
            "messages": [{"role":"system","content":system_prompt}] + [m.model_dump() for m in req.messages],
            "stream": False,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                r = await client.post(f"{OPENAI}/chat/completions", json=payload)
                rj = r.json()
                raw = rj.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
            except (httpx.RequestError, ValueError) as e:
                print(f"[router] OPENAI chat error: {e}")
                raw = ""

        content = sanitize(strip_reasoning(raw).strip()) or "인덱스에 근거 없음"
        # [변경] 폴백 시 제목 접두어 제거
        if content == "인덱스에 근거 없음" and ctx_text.strip():
            content = sanitize(ctx_text)[:600]

        # [추가] 출처 붙이기 (MCP/Confluence 경로일 때)
        if ROUTER_SHOW_SOURCES and qa_urls:
            content += "\n\n출처:\n" + "\n".join(f"- {u}" for u in qa_urls)

        return {
            "id": f"cmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{"index":0,"message":{"role":"assistant","content":content},"finish_reason":"stop"}],
        }

    # 2-B) QA 실패 → QUERY
    best_ctx_good = ""; best_ctx_any = ""
    best_urls_good: List[str] = []     # [추가]
    best_urls_any: List[str] = []      # [추가]

    async with httpx.AsyncClient(timeout=timeout) as client:
        for v in variants:
            try:
                qres = await client.post(f"{RAG}/query", json={"q": v, "k": 5})
                qj = qres.json()
            except (httpx.RequestError, ValueError) as e:
                print(f"[router] /query error for '{v}': {e}")
                continue

            # [추가] items/contexts 등에서 텍스트와 URL 모두 수집
            items = (qj.get("items") or
                     qj.get("contexts") or
                     [])
            urls = _collect_urls_from_items(items)

            ctx_list = (
                qj.get("context_texts")
                or [c.get("text","") for c in (qj.get("contexts") or [])]
                or [it.get("text","") for it in (qj.get("items") or [])]
            )
            ctx = "\n\n---\n\n".join([t for t in ctx_list if t])[:MAX_CTX_CHARS]

            if len(ctx) > len(best_ctx_any):
                best_ctx_any = ctx
                best_urls_any = urls[:]   # [추가]
            if is_good_context_for_qa(ctx) and len(ctx) > len(best_ctx_good):
                best_ctx_good = ctx
                best_urls_good = urls[:]  # [추가]

    best_ctx = best_ctx_good or best_ctx_any
    src_urls = best_urls_good or best_urls_any  # [추가]

    if not best_ctx:
        # 일반 LLM 폴백
        now_kst = datetime.now(ZoneInfo(TZ)).strftime("%Y-%m-%d (%a) %H:%M:%S %Z")
        sysmsg = {
            "role": "system",
            "content": f"현재 날짜와 시간: {now_kst}. 문서 인덱스가 없어도 일반 상식·수학·날짜/시간 등은 직접 답하세요. ‘인덱스에 근거 없음’ 같은 말은 하지 마세요."
        }
        max_tokens = req.max_tokens or ROUTER_MAX_TOKENS
        payload = {"model": OPENAI_MODEL, "messages": [sysmsg] + [m.model_dump() for m in req.messages],
                   "stream": False, "temperature": 0, "max_tokens": max_tokens}
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                r = await client.post(f"{OPENAI}/chat/completions", json=payload)
                rj = r.json()
                raw = rj.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
                content = sanitize(strip_reasoning(raw).strip()) or "죄송해요. 지금은 답을 찾지 못했어요."
            except (httpx.RequestError, ValueError):
                content = "죄송해요. 지금은 답을 찾지 못했어요."
        return {
            "id": f"cmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{"index":0,"message":{"role":"assistant","content":content},"finish_reason":"stop"}],
        }

    # QUERY 경로 LLM 호출
    ctx_text = best_ctx
    ctx_for_prompt = sanitize(ctx_text)
    system_prompt = build_system_with_context(ctx_for_prompt)
    max_tokens = req.max_tokens or ROUTER_MAX_TOKENS
    payload = {"model": OPENAI_MODEL,
               "messages":[{"role":"system","content":system_prompt}] + [m.model_dump() for m in req.messages],
               "stream": False, "temperature": 0, "max_tokens": max_tokens}
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{OPENAI}/chat/completions", json=payload)

    rj = r.json()
    raw = rj.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    content = sanitize(strip_reasoning(raw).strip()) or "인덱스에 근거 없음"

    # [변경] 폴백 시 제목 접두어 제거
    if content == "인덱스에 근거 없음" and ctx_text.strip():
        content = sanitize(ctx_text)[:600]

    # [추가] 출처 붙이기
    if ROUTER_SHOW_SOURCES and src_urls:
        content += "\n\n출처:\n" + "\n".join(f"- {u}" for u in src_urls)

    return {
        "id": f"cmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index":0,"message":{"role":"assistant","content":content},"finish_reason":"stop"}],
    }