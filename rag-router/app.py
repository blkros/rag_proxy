# rag-router/app.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import os, httpx, time, uuid, re, math, unicodedata
from html import unescape
from datetime import datetime
from zoneinfo import ZoneInfo
from functools import lru_cache

RAG = os.getenv("RAG_PROXY_URL", "http://rag-proxy:8080")
OPENAI = os.getenv("OPENAI_URL", "http://172.16.10.168:9993/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "qwen3-30b-a3b-fp8")
ROUTER_MODEL_ID = os.getenv("ROUTER_MODEL_ID", "qwen3-30b-a3b-fp8-router")
TZ = os.getenv("ROUTER_TZ", "Asia/Seoul")
_NUM_ONLY_LINE = re.compile(r'(?m)^\s*(\d{1,3}(?:,\d{3})*|\d+)\s*$')

ROUTER_MAX_TOKENS = int(os.getenv("ROUTER_MAX_TOKENS", "2048"))
ANSWER_MODE = os.getenv("ROUTER_ANSWER_MODE", "auto")
BULLETS_MAX = int(os.getenv("ROUTER_BULLETS_MAX", "15"))
MAX_CTX_CHARS = int(os.getenv("MAX_CTX_CHARS", "8000"))
_BULLET_HINTS = os.getenv(
    "ROUTER_AUTO_BULLET_HINTS",
    "정리 요약 항목 목록 리스트 불릿 bullet 체크리스트 장단점 비교 포인트 핵심 todo 해야할일"
).split()
_PARA_HINTS = os.getenv(
    "ROUTER_AUTO_PARA_HINTS",
    "설명 자세히 알려줘 소개 무엇 뭐야 해줘 왜 어떻게 의미 정의 개요 한문단 문단 서술형"
).split()

# [추가] 제목 접두어 완전 비활성(기본 ""), 필요시 환경변수로 켜도 됨
HEADING = os.getenv("ROUTER_HEADING", "")

# [추가] 출처 표시 on/off 및 최대 개수
ROUTER_SHOW_SOURCES = (os.getenv("ROUTER_SHOW_SOURCES", "1").lower() not in ("0","false","no"))
ROUTER_SOURCES_MAX  = int(os.getenv("ROUTER_SOURCES_MAX", "5"))

# === relevance gate ===
_KO_EN_TOKEN = re.compile(r"[A-Za-z0-9]+|[가-힣]{2,}")

SYNONYMS = {
    "NIA": ["한국지능정보사회진흥원", "지능정보사회진흥원", "국가정보화진흥원"],
    "상가정보": ["상권정보", "상권 분석", "상권", "상업용 부동산 정보", "상가 매물"]
}

ALIASES = {
    "NIA": ["NIA", "한국지능정보사회진흥원", "지능정보사회진흥원", "국가정보화진흥원"]
}

_STOPWORDS = set("은 는 이 가 을 를 에 의 와 과 도 로 으로 에서 에게 그리고 그러나 그래서 무엇 뭐야 뭐지 설명 해줘 대한 대해 정리 개요 소개 자세히".split())

# ===== httpx timeout/env =====
ROUTER_CONNECT_TIMEOUT = float(os.getenv("ROUTER_CONNECT_TIMEOUT", "20"))
ROUTER_READ_TIMEOUT    = float(os.getenv("ROUTER_READ_TIMEOUT", "180"))  # <- 120 → 180로 여유
ROUTER_WRITE_TIMEOUT   = float(os.getenv("ROUTER_WRITE_TIMEOUT", "60"))
ROUTER_POOL_TIMEOUT    = float(os.getenv("ROUTER_POOL_TIMEOUT", "180"))

_JOSA_RE = re.compile(r'(에서|에게|으로써|으로서|으로부터|로부터|부터|까지|만큼|보다|처럼|부터|까지|은|는|이|가|을|를|에|에서|에게|와|과|도|로|으로|의)$')

def _strip_josa(tok: str) -> str:
    return _JOSA_RE.sub('', tok)


def _httpx_timeout():
    import httpx
    return httpx.Timeout(
        connect=ROUTER_CONNECT_TIMEOUT,
        read=ROUTER_READ_TIMEOUT,
        write=ROUTER_WRITE_TIMEOUT,
        pool=ROUTER_POOL_TIMEOUT,
    )


def _spaces_from_env():
    raw = os.getenv("CONFLUENCE_SPACE", "").strip()
    if not raw:
        return None
    return [s.strip().upper() for s in raw.split(",") if s.strip()] or None

SPACES = _spaces_from_env()

# [ADD] 질문에 가장 잘 맞는 space를 자동 선택 (없으면 None 반환 → 제한 없이 진행)
async def _auto_pick_spaces(q: str, client: httpx.AsyncClient) -> list[str] | None:
    if not SPACES or len(SPACES) <= 1:
        return SPACES  # 단일 스페이스거나 미설정이면 굳이 선택 안 함

    scores = []
    try:
        # 한 번만 탐색해도 충분하니 첫 변형 쿼리로 간단 프리뷰
        probe_q = q.strip()
        for sp in SPACES:
            # 프리뷰: LLM 생성 없이 검색만(k=3) 해서 '적합도' 점수 계산
            r = await client.post(f"{RAG}/query", json={ "q": probe_q, "k": 3, "sticky": False , "spaces": [sp] })

            j = r.json()
            # 컨텍스트 텍스트를 조금 모아서 질문과의 토큰 겹침(relevance_ratio)로 스코어링
            ctx = "\n\n".join(extract_texts(j.get("items") or j.get("contexts") or []))[:1500]
            score = float(j.get("hits") or 0) + relevance_ratio(probe_q, ctx)  # 간단 복합 점수
            scores.append((score, sp))
    except Exception as e:
        print(f"[router] space probe failed: {e}")

    # 최고 점수만 채택 (0 이하면 제한 하지 않음)
    if not scores:
        return None
    scores.sort(key=lambda x: x[0], reverse=True)
    top = scores[0]
    # 동률이거나 근소 차이면 제한하지 않고(None) 진행 → 오탐 방지
    if top[0] > 0 and (len(scores) == 1 or top[0] >= scores[1][0] + 0.2):
        return [top[1]]
    return None

def _tokens(s: str) -> list[str]:
    toks = []
    for t in _KO_EN_TOKEN.findall(s or ""):
        t2 = _strip_josa(t).lower()
        if t2:
            toks.append(t2)
    return toks


def relevance_ratio(q: str, ctx: str, ctx_limit: int = 2000) -> float:
    qk = [t for t in _tokens(q) if t not in _STOPWORDS]
    if not qk:
        return 0.0

    # [추가] 공백 제거 후 부분문자열 매칭(한글 합성어 대응)
    qnorm = re.sub(r'\s+', '', _normalize_query(q).lower())
    cnorm = re.sub(r'\s+', '', (ctx or "")[:ctx_limit].lower())
    if len(qnorm) >= 4 and qnorm in cnorm:
        return 1.0

    ck = set(_tokens((ctx or "")[:ctx_limit]))
    common = sum(1 for t in qk if t in ck)
    return common / len(qk)


# 환경변수로 문턱값 조정 가능 (기본 0.2)
REL_THRESH = float(os.getenv("ROUTER_MIN_OVERLAP", "0.08"))

def _normalize_query(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_relevant(q: str, ctx: str) -> bool:
    return relevance_ratio(q, ctx) >= REL_THRESH

def _is_webui_task(s: str) -> bool:
    return bool(re.match(r"(?is)^\s*#{3}\s*task\s*:", (s or "")))

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
    if not text:
        return text
    # assistant_thought 블록 제거 (response 토큰 전까지)
    text = re.sub(r'(?is)<\|assistant_thought\|>.*?(?=<\|assistant_response\|>|\Z)', '', text)
    # response 토큰 마커 제거
    text = re.sub(r'(?is)<\|assistant_response\|>', '', text)

    # <think>…</think> 또는 </think> 없이 끝까지
    text = re.sub(r'(?is)<think\b[^>]*>.*?(?:</think>|$)', '', text)

    # 코드블록 형태의 생각/추론
    text = re.sub(r'(?is)```(?:thinking|reasoning|thought|scratchpad)[\s\S]*?```', '', text)

    # “Thought:” “Reasoning:” 스타일(빈 줄까지)
    text = re.sub(r'(?im)^\s*(?:thought|reasoning|scratchpad)\s*:\s*[\s\S]*?(?:\n{2,}|\Z)', '', text)

    return text.strip()


def mark_lonely_numbers_as_total(text: str) -> str:
    """
    줄 전체가 숫자만으로 이루어진 경우 '(합계: N)'으로 바꿔
    LLM이 개별 항목 수치로 오해하지 않도록 힌트를 준다.
    """
    def repl(m: re.Match):
        n = m.group(1)
        return f"(합계: {n})"
    return _NUM_ONLY_LINE.sub(repl, text)

# [추가] 컨텍스트가 '목록스러움'을 보이는지 가볍게 스코어링
def _looks_structured(ctx: str) -> bool:
    if not ctx: return False
    lines = [ln.strip() for ln in ctx.splitlines() if ln.strip()]
    if len(lines) < 4:
        return False
    bullet_like = 0
    for ln in lines[:40]:
        if re.match(r"^(?:[-•*]\s+|\d+\.\s+|\[\w+\]\s+)", ln):
            bullet_like += 1
        elif len(ln) <= 28:
            bullet_like += 0.5
    return bullet_like >= 4

def pick_answer_mode(user_msg: str, ctx_text: str) -> str:
    if ANSWER_MODE != "auto":
        return ANSWER_MODE
    um = (user_msg or "").lower()
    if any(k.lower() in um for k in _BULLET_HINTS):
        return "bulleted"
    if any(k.lower() in um for k in _PARA_HINTS):
        return "paragraph"
    return "bulleted" if _looks_structured(ctx_text) else "paragraph"


# --- utils ----------------------------------------------------

def normalize_query(q: str) -> str:
    if not q: return ""
    s = q.strip()
    # s = re.sub(r'(?i)\bstep\b', 'SFTP', s)
    s = re.sub(r'(?i)\bstfp\b|\bsfttp\b|\bsfpt\b|\bsftp\b', 'SFTP', s)
    s = s.replace("스텝", "SFTP")
    return s

def _expand_synonyms(s: str) -> list[str]:
    out = [s]
    for k, vs in SYNONYMS.items():
        if k in s:
            for v in vs:
                out.append(s.replace(k, v))
    ss = s.replace("상가 정보", "상가정보")
    if ss != s: out.append(ss)
    return list(dict.fromkeys(out))

def generate_query_variants(q: str, limit: int = 12) -> List[str]:
    s = normalize_query(q)
    cand = []
    def add(x): 
        x = re.sub(r'\s+',' ',x).strip()
        if x and x not in cand: cand.append(x)
    add(s); add(re.sub(r'\s+','',s))
    add(re.sub(r'([가-힣])([A-Za-z0-9])', r'\1 \2', s))
    add(re.sub(r'([A-Za-z0-9])([가-힣])', r'\1 \2', s))
    for v in _expand_synonyms(s):
        add(v); add(re.sub(r'\s+','',v))
    # 기존 pairs 유지
    return cand[:limit]

def sanitize(text: str) -> str:
    if not text: return ""
    t = unescape(text.replace("&nbsp;", " "))
    t = re.sub(r'(?i)(password|passwd|pwd|패스워드|비밀번호)\s*[:=]\s*\S+', r'\1: ******', t)
    t = re.sub(r'(?i)(token|secret|key|키)\s*[:=]\s*[A-Za-z0-9\-_]{6,}', r'\1: <redacted>', t)
    t = re.sub(r'(?i)(account|user(?:name)?|userid|계정|아이디)\s*[:=]\s*\S+', r'\1: <redacted>', t)
    t = re.sub(r'\b(\d{1,3}\.\d{1,3}\.\d{1,3})\.\d{1,3}\b', r'\1.xxx', t)
    return t

def build_system_with_context(ctx_text: str, mode: str) -> str:
    if mode == "bulleted":
        style = (
            f"- 최대 {BULLETS_MAX}개 불릿으로 **구체적**으로 서술한다.\n"
            "- 각 불릿은 2~4문장으로 쓴다.\n"
            "- 불릿 외의 군더더기 서론/결론 문단은 길게 넣지 않는다.\n"
        )
    elif mode == "sections":
        style = (
            "- 2~4개의 **문단**으로 핵심→배경→세부→시사점 순으로 정리한다.\n"
            "- 마크다운 리스트 문법은 사용하지 않는다.\n"
        )
    else:
        style = (
            "- **리스트/번호/하이픈(-, •, 1.) 없이** 한두 개의 **연속된 문단**으로 자연스럽게 작성한다.\n"
            "- 첫 문장에 요지를 분명히 말하고, 이어서 구성요소·동작·제약을 설명한다.\n"
        )

    numeric_rules = (
        "- 표/목록의 **수치**는 **같은 행(같은 항목)** 에 적힌 숫자만 인용한다.\n"
        "- **합계/총계** 숫자를 개별 항목 값으로 배정하지 않는다.\n"
        "- 숫자를 쓸 때는 반드시 `항목명 숫자`로 **쌍**을 이뤄 서술한다. (예: `반포동 47`)\n"
        "- 상위 단위 합계는 필요 시 `(서초구 합계 439)`처럼 **합계임을 명시**한다.\n"
        "- 불명확하면 숫자 대신 '수치 불분명'으로 적는다.\n"
    )
    heading_hint = (f"- 가능하면 '{HEADING}' 아래로 정리한다.\n" if HEADING else "")

    return (
        "역할: 주어진 컨텍스트를 근거로 **정확하고 실무 친화적인** 한국어 답변을 작성한다.\n"
        "원칙:\n"
        "- 컨텍스트에 있는 정보만 사용하고 추측 금지.\n"
        "- 고유명사/수치는 가능한 그대로 인용하되 과도한 반복은 피한다.\n"
        "- 내부 추론(<think> 등) 출력 금지, 최종 답만 출력한다.\n"
        + heading_hint + style + numeric_rules +
        "- 컨텍스트가 완전히 비었거나 무관하면 정확히 `인덱스에 근거 없음`만 출력한다.\n"
        "- 민감정보(비밀번호/토큰/IP 마지막 옥텟)는 마스킹한다.\n"
        "[컨텍스트 시작]\n"
        f"{ctx_text}\n"
        "[컨텍스트 끝]\n"
    )

def _limit_urls(urls: List[str] | None, top_n: int = ROUTER_SOURCES_MAX) -> List[str]:
    out, seen = [], set()
    for u in urls or []:
        nu = _normalize_url(u)
        if nu and nu not in seen:
            seen.add(nu); out.append(nu)
        if len(out) >= top_n:
            break
    return out


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

# [추가] URL 정규화(Confluence pageId 기준으로 중복 제거)
def _normalize_url(u: str) -> str:
    if not u:
        return ""
    u = str(u).split("#")[0].strip().rstrip("/")
    m = re.search(r"(pageId=\d+)", u)
    if m:
        base = u.split("?")[0]
        return f"{base}?{m.group(1)}"
    return u

def _collect_urls_from_items(items: List[dict], top_n: Optional[int] = None) -> List[str]:
    top_n = top_n or ROUTER_SOURCES_MAX
    cands = []

    def push(it: dict):
        if not isinstance(it, dict):
            return
        score = float(it.get("score") or it.get("similarity") or 0.0)
        url = it.get("url") or it.get("source_url") or it.get("link")
        if url:
            cands.append((score, _normalize_url(str(url))))
        payload = it.get("payload") or it.get("data") or {}
        if isinstance(payload, dict):
            url2 = payload.get("url") or payload.get("source_url") or payload.get("link")
            if url2:
                cands.append((score, _normalize_url(str(url2))))
        meta = it.get("metadata") or {}
        if isinstance(meta, dict):
            url3 = meta.get("url")
            if url3:
                cands.append((score, _normalize_url(str(url3))))
            if not (url or (payload if isinstance(payload, dict) else {}).get("url") or url3):
                src = meta.get("source")
                if src:
                    cands.append((score, str(src)))

    for it in items or []:
        push(it)

    cands = [(s, u) for (s, u) in cands if u]
    cands.sort(key=lambda x: x[0], reverse=True)

    out: List[str] = []
    for _, u in cands:
        if u not in out:
            out.append(u)
        if len(out) >= top_n:
            break
    return out

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

    # 메타 태스크면 RAG 건너뛰고 그대로 모델로 전달 (JSON 형식 보존)
    if _is_webui_task(orig_user_msg):
        payload = {
            "model": OPENAI_MODEL,
            "messages": [m.model_dump() for m in req.messages],
            "stream": False,
            "temperature": 0,
            "max_tokens": req.max_tokens or ROUTER_MAX_TOKENS,
        }
        try:
            async with httpx.AsyncClient(timeout=_httpx_timeout()) as client:
                r = await client.post(f"{OPENAI}/chat/completions", json=payload)
                j = r.json()

                # ★ 추가: 메타 태스크 응답도 think 제거
                try:
                    msg = j.get("choices", [{}])[0].get("message", {})
                    c = msg.get("content") or ""
                    msg["content"] = strip_reasoning(c)
                    j["choices"][0]["message"] = msg
                except Exception:
                    pass

                return j
        except (httpx.RequestError, ValueError) as e:
            # 타임아웃/네트워크 장애는 200으로 안전하게 래핑해 돌려줌
            return {
                "id": f"cmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "메타 태스크 처리 중 백엔드 응답 지연으로 실패했어요. 잠시 후 다시 시도해주세요."
                    },
                    "finish_reason": "stop"
                }],
                "error": {"type": e.__class__.__name__, "message": str(e)}
            }


    ctx_text = ""
    qa_json = None
    qa_items = []
    qa_urls: List[str] = []     # QA 경로 출처

    timeout = _httpx_timeout()
    async with httpx.AsyncClient(timeout=timeout) as client:
        spaces_hint = await _auto_pick_spaces(orig_user_msg, client)

        # ====== 2-A) QA 경로 시도 ======
        qa_json = None; qa_items = []; qa_urls = []

        for v in variants:
            # 1) 글로벌(Confluence 등) 검색: sticky False
            try:
                j1 = (await client.post(f"{RAG}/qa", json={"q": v, "k": 5, "sticky": False})).json()
            except Exception as e:
                print(f"[router] /qa global error for '{v}': {e}")
                j1 = {}

            # 2) PDF(세션 sticky) 검색: sticky True   ※ /qa에는 spaces 절대 넣지 않음
            try:
                j2 = (await client.post(f"{RAG}/qa", json={"q": v, "k": 5, "sticky": True})).json()
            except Exception as e:
                print(f"[router] /qa pdf error for '{v}': {e}")
                j2 = {}

            # 두 결과 중 더 나은 쪽을 선택
            cand = None
            for jj in (j1, j2):
                if (jj.get("hits") or 0) > 0:
                    cand = jj; break
            if not cand:
                continue

            qa_json = cand
            qa_items = cand.get("items", [])
            qa_urls  = (_limit_urls(cand.get("source_urls"))
                        if cand.get("source_urls") else _collect_urls_from_items(qa_items, top_n=ROUTER_SOURCES_MAX))
            break


    # 2-A) QA 성공
    if qa_json:
        ctx_text = "\n\n".join(extract_texts(qa_items))[:MAX_CTX_CHARS]
        ctx_text = mark_lonely_numbers_as_total(ctx_text)
    # [CHANGE] 길이(80자) 허용 삭제 → 관련도/컨텍스트 품질만
    qa_ok = bool(ctx_text.strip()) and (is_good_context_for_qa(ctx_text) or is_relevant(orig_user_msg, ctx_text))
    if not qa_ok:
        qa_json = None

    if qa_json:
        ctx_for_prompt = sanitize(ctx_text)    
        mode = pick_answer_mode(orig_user_msg, ctx_for_prompt)
        system_prompt = build_system_with_context(ctx_for_prompt, mode)
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
        # if content == "인덱스에 근거 없음" and ctx_text.strip():
        #     content = sanitize(ctx_text)[:600]

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

    # ====== 2-B) QA 실패 → QUERY 경로 ======
    async with httpx.AsyncClient(timeout=timeout) as client:
        best_ctx_good=""; best_ctx_any=""
        best_urls_good=[]; best_urls_any=[]
        used_q_good=None; used_q_any=None

        for v in variants:
            # 1) 글로벌
            try:
                payload = {"q": v, "k": 5, "sticky": False}
                if spaces_hint: payload["spaces"] = spaces_hint  # QUERY엔 spaces OK
                qj1 = (await client.post(f"{RAG}/query", json=payload)).json()
            except Exception as e:
                print(f"[router] /query global error for '{v}': {e}")
                qj1 = {}

            # 2) PDF
            try:
                qj2 = (await client.post(f"{RAG}/query", json={"q": v, "k": 5, "sticky": True})).json()
            except Exception as e:
                print(f"[router] /query pdf error for '{v}': {e}")
                qj2 = {}

            # ★ 두 후보를 모두 평가하면서 best 갱신 (여기서 끝!)
            for qj in (qj1, qj2):
                items = (qj.get("items") or qj.get("contexts") or [])
                urls  = (_limit_urls(qj.get("source_urls"))
                        if qj.get("source_urls") else _collect_urls_from_items(items, top_n=ROUTER_SOURCES_MAX))
                ctx_list = (qj.get("context_texts")
                            or [c.get("text","") for c in (qj.get("contexts") or [])]
                            or [it.get("text","") for it in (qj.get("items") or [])])
                ctx = "\n\n---\n\n".join([t for t in ctx_list if t])[:MAX_CTX_CHARS]

                if len(ctx) > len(best_ctx_any):
                    best_ctx_any = ctx
                    best_urls_any = urls[:]
                    used_q_any = v
                if is_good_context_for_qa(ctx) and len(ctx) > len(best_ctx_good):
                    best_ctx_good = ctx
                    best_urls_good = urls[:]
                    used_q_good = v


        best_ctx = best_ctx_good or best_ctx_any
        src_urls = best_urls_good or best_urls_any
        used_q_for_relevance = used_q_good if best_ctx_good else used_q_any

        # 게이트
        if best_ctx and not (is_good_context_for_qa(best_ctx) or
                            is_relevant(used_q_for_relevance or orig_user_msg, best_ctx)):
            best_ctx = ""
            src_urls = []


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
    ctx_text = mark_lonely_numbers_as_total(ctx_text) 
    ctx_for_prompt = sanitize(ctx_text)
    mode = pick_answer_mode(orig_user_msg, ctx_for_prompt)
    system_prompt = build_system_with_context(ctx_for_prompt, mode)
    max_tokens = req.max_tokens or ROUTER_MAX_TOKENS
    payload = {
        "model": OPENAI_MODEL,
        "messages":[{"role":"system","content":system_prompt}] + [m.model_dump() for m in req.messages],
        "stream": False, "temperature": 0, "max_tokens": max_tokens
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
    # if content == "인덱스에 근거 없음" and ctx_text.strip():
    #     content = sanitize(ctx_text)[:600]

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