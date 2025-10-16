from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import os, httpx, time, uuid, re
from html import unescape  # ← (추가) HTML 엔티티 정리용
from datetime import datetime
from zoneinfo import ZoneInfo

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

# --- utils: query normalizer & sanitizer ---

def normalize_query(q: str) -> str:
    """
    사용자가 'STEP'이라고 잘못 적은 걸 'SFTP'로 보정하고,
    자주 나오는 철자 실수(sfttp, stfp 등)도 SFTP로 정규화.
    """
    if not q:
        return ""
    s = q.strip()

    # 'STEP' -> 'SFTP'
    s = re.sub(r'(?i)\bstep\b', 'SFTP', s)
    # 흔한 오타 보정 (stfp, sfttp 등)
    s = re.sub(r'(?i)\bstfp\b|\bsfttp\b|\bsfpt\b|\bsftp\b', 'SFTP', s)
    # 한국어 표기 흔적
    s = s.replace("스텝", "SFTP")

    return s

def generate_query_variants(q: str, limit: int = 6) -> List[str]:
    """
    띄어쓰기/한영 경계/자주 섞어 쓰는 합성어(개발서버/개발 서버 등) 변형을 만들어
    /qa → /query 순으로 시도할 때 히트율을 올린다.
    """
    s = normalize_query(q)
    cand: List[str] = []

    def add(x: str):
        x = re.sub(r'\s+', ' ', x).strip()
        if x and x not in cand:
            cand.append(x)

    # 기본형
    add(s)
    # 공백 제거형 (예: "개발 서버 정보" → "개발서버정보")
    add(re.sub(r'\s+', '', s))
    # 한글-영문/숫자 경계에 공백 삽입
    add(re.sub(r'([가-힣])([A-Za-z0-9])', r'\1 \2', s))
    add(re.sub(r'([A-Za-z0-9])([가-힣])', r'\1 \2', s))

    # 자주 헷갈리는 합성어 ↔ 분리어 쌍 (양방향 모두 추가)
    pairs = [
        ("개발 서버", "개발서버"),
        ("테스트 서버", "테스트서버"),
        ("운영 서버", "운영서버"),
        ("계정 정보", "계정정보"),
        ("접속 정보", "접속정보"),
        ("IP 주소", "IP주소"),
    ]
    for a, b in pairs:
        add(s.replace(a, b))
        add(s.replace(b, a))

    return cand[:limit]

def sanitize(text: str) -> str:
    if not text:
        return ""
    t = unescape(text.replace("&nbsp;", " "))

    # password / 패스워드 / 비밀번호  ← [수정] '비밀번호' 추가
    t = re.sub(r'(?i)(password|passwd|pwd|패스워드|비밀번호)\s*[:=]\s*\S+', r'\1: ******', t)

    # token / key
    t = re.sub(r'(?i)(token|secret|key|키)\s*[:=]\s*[A-Za-z0-9\-_]{6,}', r'\1: <redacted>', t)

    # account / username
    t = re.sub(r'(?i)(account|user(?:name)?|userid|계정|아이디)\s*[:=]\s*\S+', r'\1: <redacted>', t)

    # IPv4 마지막 옥텟 마스킹
    t = re.sub(r'\b(\d{1,3}\.\d{1,3}\.\d{1,3})\.\d{1,3}\b', r'\1.xxx', t)
    return t


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
        "- 민감정보(비밀번호/토큰/IP 마지막 옥텟 등)는 반드시 마스킹한 뒤 답한다.\n"
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
    variants = generate_query_variants(orig_user_msg)

    ctx_text = ""  # [FIX-1] 어떤 코드 경로에서도 참조 가능하도록 기본값으로 초기화

    # 1) /qa 변형들 순차 시도
    qa_json = None
    qa_items = []
    # (timeout 세분화: read만 좀 더 길게)
    timeout = httpx.Timeout(30.0, connect=5.0, read=20.0, write=20.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for v in variants:
            try:  # ← [추가] 예외 흡수해서 다음 변형 시도
                qa = await client.post(f"{RAG}/qa", json={"q": v, "k": 5})
                j = qa.json()
            except (httpx.RequestError, ValueError) as e:
                print(f"[router] /qa error for '{v}': {e}")  # ← 로그만 찍고 계속
                continue

            if (j.get("hits") or 0) > 0:
                qa_json = j
                qa_items = j.get("items", [])
                break


    # 2-A) /qa에서 뭔가 나오면: 컨텍스트로 LLM 호출
    if qa_json:
        ctx_text = "\n\n".join(extract_texts(qa_items))[:8000]
        # (선택) 너무 빈약하면 /query로 강등
        if not is_good_context_for_qa(ctx_text):
            qa_json = None  # 아래 2-B 경로로

    if qa_json:
        ctx_for_prompt = sanitize(ctx_text)
        system_prompt = build_system_with_context(ctx_for_prompt)
        payload = {
            "model": OPENAI_MODEL,
            "messages": [{"role": "system", "content": system_prompt}] + [m.model_dump() for m in req.messages],
            "stream": False,
            "temperature": 0,
        }
        # (qa 경로)
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                r = await client.post(f"{OPENAI}/chat/completions", json=payload)
                rj = r.json()
                raw = rj.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
            except (httpx.RequestError, ValueError) as e:
                print(f"[router] OPENAI chat error: {e}")
                raw = ""  # 아래 공통 처리로 폴백

        content = strip_reasoning(raw).strip() or "인덱스에 근거 없음"
        content = sanitize(content)
        if content == "인덱스에 근거 없음" and ctx_text.strip():
            content = "컨텍스트에서 확인된 항목:\n" + sanitize(ctx_text)[:600]

        return {
            "id": f"cmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        }

    # 2-B) /qa 전부 실패 → /query 변형들 중 '가장 많은 컨텍스트' 선택
    best_ctx_good = ""
    best_ctx_any = ""
    async with httpx.AsyncClient(timeout=timeout) as client:
        for v in variants:
            try:  # ← [추가]
                qres = await client.post(f"{RAG}/query", json={"q": v, "k": 5})
                qj = qres.json()
            except (httpx.RequestError, ValueError) as e:
                print(f"[router] /query error for '{v}': {e}")
                continue

            ctx_list = (
                qj.get("context_texts")
                or [c.get("text", "") for c in (qj.get("contexts") or [])]
                or [it.get("text", "") for it in (qj.get("items") or [])]
            )
            ctx = "\n\n---\n\n".join([t for t in ctx_list if t])[:8000]
            if len(ctx) > len(best_ctx_any):
                best_ctx_any = ctx
            if is_good_context_for_qa(ctx) and len(ctx) > len(best_ctx_good):
                best_ctx_good = ctx

    best_ctx = best_ctx_good or best_ctx_any

    if not best_ctx:
        # LLM 폴백 (일반 질문/날짜/상식 응답)
        now_kst = datetime.now(ZoneInfo(TZ)).strftime("%Y-%m-%d (%a) %H:%M:%S %Z")
        sysmsg = {
            "role": "system",
            "content": (
                f"현재 날짜와 시간: {now_kst}. "
                "문서 인덱스가 없어도 일반 상식·수학·날짜/시간 등은 직접 답하세요. "
                "‘인덱스에 근거 없음’ 같은 말은 하지 마세요."
            ),
        }
        payload = {
            "model": OPENAI_MODEL,
            "messages": [sysmsg] + [m.model_dump() for m in req.messages],
            "stream": False,
            "temperature": 0,
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                r = await client.post(f"{OPENAI}/chat/completions", json=payload)
                rj = r.json()
                raw = rj.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
                content = sanitize(strip_reasoning(raw).strip()) or "죄송해요. 지금은 답을 찾지 못했어요."
            except (httpx.RequestError, ValueError) as e:
                print(f"[router] OPENAI fallback error: {e}")
                content = "죄송해요. 지금은 답을 찾지 못했어요."

        return {
            "id": f"cmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        }


    ctx_text = best_ctx
    ctx_for_prompt = sanitize(ctx_text)
    system_prompt = build_system_with_context(ctx_for_prompt)
    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "system", "content": system_prompt}] + [m.model_dump() for m in req.messages],
        "stream": False,
        "temperature": 0,
    }
    # 여기도 timeout=None -> timeout=timeout
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{OPENAI}/chat/completions", json=payload)

    rj = r.json()
    raw = rj.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    content = strip_reasoning(raw).strip() or "인덱스에 근거 없음"
    content = sanitize(content)
    if content == "인덱스에 근거 없음" and ctx_text.strip():
        content = "컨텍스트에서 확인된 항목:\n" + sanitize(ctx_text)[:600]

    return {
        "id": f"cmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
    }