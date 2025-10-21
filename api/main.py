# C:\Users\nuri\Desktop\RAG\ai-stack\api\main.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import shutil, os, logging, re, uuid
from fastapi.responses import RedirectResponse
from fastapi import Body
import time
import traceback
import src.vectorstore as VS
from pydantic import BaseModel
from datetime import datetime
from zoneinfo import ZoneInfo
import inspect
import unicodedata

from src.utils import proxy_get, call_chat_completions, drop_think
from src.rag_pipeline import build_rag_chain, Document
from src.loaders import load_docs_any
from fastapi.middleware.cors import CORSMiddleware
from src.config import settings
from src.ext.confluence_mcp import mcp_search
import asyncio, hashlib
from collections import Counter
from src.retrieval.rerank import parse_query_intent, pick_for_injection
from api.smart_router import router as smart_router

# empty FAISS 빌드를 위한 보조들
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS as FAISSStore
from langchain_community.docstore.in_memory import InMemoryDocstore

logging.basicConfig(level=logging.INFO)
DISABLE_INTERNAL_MCP = (os.getenv("DISABLE_INTERNAL_MCP", "0").lower() in ("1","true","yes"))

PAGE_FILTER_MODE = getattr(settings, "PAGE_FILTER_MODE", "soft")   # "soft"|"hard"|"off"
PAGE_HINT_BONUS = float(getattr(settings, "PAGE_HINT_BONUS", 0.35))
SINGLE_SOURCE_COALESCE = bool(getattr(settings, "SINGLE_SOURCE_COALESCE", True))
COALESCE_THRESHOLD = float(getattr(settings, "COALESCE_THRESHOLD", 0.55))
_COALESCE_TRIGGER = re.compile(r"(최근|이슈|목록|정리|요약|top\s*\d+|\d+\s*가지)", re.I)

SPACE_FILTER_MODE = getattr(settings, "SPACE_FILTER_MODE", "hard")
SPACE_HINT_BONUS = float(getattr(settings, "SPACE_HINT_BONUS", 0.25))
TITLE_BONUS      = float(getattr(settings, "TITLE_BONUS", 0.20))
ENABLE_SPARSE    = bool(getattr(settings, "ENABLE_SPARSE", False))
SPARSE_LIMIT     = int(getattr(settings, "SPARSE_LIMIT", 150))

TZ_NAME = getattr(settings, "TZ_NAME", "Asia/Seoul")

LOCAL_FIRST = bool(getattr(settings, "LOCAL_FIRST", True))
LOCAL_BONUS = float(getattr(settings, "LOCAL_BONUS", 0.25))

META_PAT  = re.compile(r"(주제|개요|요약|무엇|무슨\s*내용)", re.I)
TITLE_PAT = re.compile(r"(제목|title|문서명|파일명)", re.I)
LOGIN_PAT = re.compile(r"(Confluence에\s*로그인|로그인\s*-\s*Confluence|name=[\"']os_username[\"'])", re.I)
THIS_FILE_PAT = re.compile(
    r"(이\s*(?:pdf|엑셀|hwp|한글|워드|ppt|파워포인트)?\s*(?:파일|문서|자료)|"
    r"해당\s*(?:파일|문서|자료|pdf)|"
    r"첨부(?:한)?\s*(?:파일|문서|자료|pdf)|"
    r"방금\s*(?:올린|업로드한)\s*(?:파일|문서|자료|pdf)|"
    r"위\s*(?:파일|문서|자료))",
    re.I
)
_CHAPTER_RE = re.compile(r"제\s*(\d+)\s*장")
_ARTICLE_RE = re.compile(r"제\s*(\d+)\s*조")

last_source: Optional[str] = None
# >>> [ADD] 최근 소스 잠금 상태 (연속 질문 안정화용)
current_source: Optional[str] = None
current_source_until: float = 0.0
STICKY_SECS = int(getattr(settings, "STICKY_SECS", 180))

STICKY_STRICT = bool(getattr(settings, "STICKY_STRICT", True))
STICKY_FROM_COALESCE = bool(getattr(settings, "STICKY_FROM_COALESCE", False))  # 기본 False
STICKY_AFTER_MCP = bool(getattr(settings, "STICKY_AFTER_MCP", False))          # 기본 False

# ← [ADD] 앵커 추출(질문 핵심어) + sticky 유효성 검사
_GENERIC = set("보고 보고서 리포트 정보 정리 페이지 자료 문서 요약 정책 통계 항목 사이트 url 링크 출처".split())

# >>> [ADD] pageId 추출/URL 정규화 유틸
_PAGEID_RE = re.compile(r"[?&]pageId=(\d+)")


# 1) '오늘/지금/현재' 같은 지시어가 필수
_DEICTIC_RE = re.compile(r"(오늘|지금|현재)", re.I)

# 2) 사용자가 특정 날짜를 콕 집은 경우는 RAG로(예: 2025년 7월 3일)
_EXPLICIT_DATE_RE = re.compile(r"\d{4}\s*년|\d{1,2}\s*월\s*\d{1,2}\s*일", re.I)

# 3) 리스트/요약/탑N 같은 RAG 냄새 신호가 있으면 RAG로
_RETRIEVAL_HINT_RE = re.compile(r"(최근|이슈|목록|리스트|정리|요약|top\s*\d+|\d+\s*가지|내역|중)", re.I)

# 4) 날짜/시간을 '오늘' 기준으로 물어보는지
_DATE_TIME_NEED_RE = re.compile(r"(날짜|요일|시간|시각|몇\s*시|몇\s*분|몇\s*월|몇\s*일|며칠)", re.I)

_DOMAIN_HINT_RE = re.compile(r"(회의|마감|일정|보고서|티켓|이슈|장애|배포|회의록|결재|승인|요청|문서)", re.I)

# 맨 위 import들 아래 어딘가
def _spaces_from_env():
    raw = os.getenv("CONFLUENCE_SPACE", "").strip()
    if not raw:
        return None
    return [s.strip().upper() for s in raw.split(",") if s.strip()] or None

ENV_SPACES = _spaces_from_env()

def _resolve_allowed_spaces(client_spaces: list | None) -> list | None:
    """
    ENV_SPACES가 있으면 그게 '최대 허용치'.
    클라이언트가 spaces를 보냈으면 교집합만 허용.
    둘 다 없으면 제한 없음(None).
    """
    cs = [s.strip().upper() for s in (client_spaces or []) if isinstance(s, str) and s.strip()]
    if ENV_SPACES and cs:
        inter = [s for s in cs if s in ENV_SPACES]
        return inter or ENV_SPACES
    return ENV_SPACES or (cs or None)

def _mcp_results_to_items(mcp_results: list[dict], k: int) -> tuple[list[dict], list[dict], list[Document]]:
    items, contexts, up_docs = [], [], []
    for r in (mcp_results or [])[:k]:
        title = (r.get("title") or "").strip()
        body  = (r.get("body") or r.get("excerpt") or r.get("text") or "")
        body  = re.sub(r"@@@(?:hl|endhl)@@@", "", body)
        if not (title or body.strip()):
            continue

        # pageId 보강
        pid = (r.get("id") or "").strip()
        if not pid:
            m_pid = re.search(r"[?&]pageId=(\d+)", (r.get("url") or ""))
            if m_pid: pid = m_pid.group(1)

        src   = (r.get("url") or f"confluence:{pid}").strip()
        space = (r.get("space") or "").strip()
        text  = ((title + "\n\n") if title else "") + body

        md = {
            "source": src,
            "url": src,
            "kind": "confluence",
            "page": pid or None,
            "space": space,
            "title": title,
        }
        items.append({"text": text, "metadata": md, "score": 0.5})
        contexts.append({"text": text, "source": md["source"], "page": md["page"], "kind": md["kind"], "score": 0.5})

        # 인덱스 업서트용
        if title:
            up_docs.append(Document(page_content=f"[TITLE] {title}",
                                    metadata={"source": src, "title": title, "space": space, "kind": "title", "page": pid or None}))
        for c in _chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP):
            up_docs.append(Document(page_content=c,
                                    metadata={"source": src, "title": title, "space": space, "kind": "chunk", "page": pid or None}))
    return items, contexts, up_docs


def _anchor_tokens_from_query(q: str) -> list[str]:
    # _tokenize_query는 파일에 이미 정의되어 있음
    toks = _tokenize_query(q)
    return [t for t in toks if len(t) >= 3 and t not in _GENERIC]

def _collect_text_of_source(src: str, limit_chars: int = 2000) -> str:
    """docstore에서 해당 source 텍스트/제목을 조금 모아 샘플링"""
    try:
        ds = getattr(vectorstore, "docstore", None)
        dct = getattr(ds, "_dict", {}) if ds else {}
    except Exception:
        dct = {}
    norm = _norm_source(src)
    buf_title, buf_body = [], []
    for d in dct.values():
        md = d.metadata or {}
        if _norm_source(str(md.get("source",""))) != norm:
            continue
        if md.get("title"):
            buf_title.append(str(md.get("title")))
        if d.page_content:
            buf_body.append(d.page_content)
        if sum(len(x) for x in buf_body) > limit_chars:
            break
    title_part = " ".join(buf_title)
    body_part  = " ".join(buf_body)[:limit_chars]
    return (title_part + "\n" + body_part).strip()

def _sticky_is_relevant(q: str, src: str) -> bool:
    anchors = _anchor_tokens_from_query(q)
    if not anchors:
        return True  # 앵커가 없으면 관대하게 허용
    sample = _collect_text_of_source(src, limit_chars=3000).lower()
    if not sample:
        return False
    hit = sum(1 for a in anchors if a.lower() in sample)
    need = 1 if len(anchors) <= 2 else max(2, int(len(anchors)*0.5))
    return hit >= need

def _is_datetime_question(q: str) -> bool:
    q = (q or "").strip()
    if not q:
        return False
    # RAG 신호가 있으면 직접답 차단
    if _EXPLICIT_DATE_RE.search(q) or _RETRIEVAL_HINT_RE.search(q):
        return False
    # '오늘/지금/현재' 같은 지시어가 반드시 있어야 함
    if not _DEICTIC_RE.search(q):
        return False
    if _DOMAIN_HINT_RE.search(q):
        return False
    # 실제로 날짜/시간을 묻는지 확인
    return bool(_DATE_TIME_NEED_RE.search(q))

def _now_str_kst() -> tuple[str, str, str]:
    try:
        now = datetime.now(ZoneInfo(TZ_NAME))
    except Exception:
        # tzdata 미설치 등 문제 시 로컬 타임 폴백
        now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    dow_str  = ["월","화","수","목","금","토","일"][now.weekday()]
    time_str = now.strftime("%H:%M")
    return date_str, dow_str, time_str

async def _call_llm(messages: list[dict], **kwargs) -> str:
    """call_chat_completions이 sync/async 어느 쪽이든 안전하게 호출"""
    try:
        if inspect.iscoroutinefunction(call_chat_completions):
            return await call_chat_completions(messages=messages, **kwargs)
        return call_chat_completions(messages=messages, **kwargs)
    except TypeError:
        # 혹시 위치인자만 받는 구현일 수도 있어서 백업
        if inspect.iscoroutinefunction(call_chat_completions):
            return await call_chat_completions(messages)
        return call_chat_completions(messages)

def _extract_page_id(src: Optional[str]) -> Optional[str]:
    """Confluence URL 에서 pageId= 숫자만 뽑는다."""
    if not src:
        return None
    m = _PAGEID_RE.search(str(src))
    return m.group(1) if m else None

def _url_has_page_id(url: Optional[str], page_id: Optional[str]) -> bool:
    """url 이 해당 pageId 를 가리키는지 여부"""
    if not url or not page_id:
        return False
    return bool(_PAGEID_RE.search(str(url)) and page_id in str(url))

def _set_sticky(src: str, secs: int = STICKY_SECS):
    """해당 소스를 잠시 기본 대상으로 고정(연속 질문 시 섞임 방지)"""
    global current_source, current_source_until
    current_source = _norm_source(src)
    current_source_until = time.time() + secs

def _norm_source(s: str) -> str:
    s = (s or "").replace("\\", "/")
    return re.sub(r"^/app/", "", s)  # 컨테이너 절대경로를 제거해 'uploads/…'로 맞춤

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _norm_kr(s: str) -> str:
    # 공백 제거 + 도메인 동의어/정규화 적용 + 소문자
    t = _basic_normalize(s)
    t = _apply_canon_map(t)
    t = re.sub(r"\s+", "", t)        # 공백 제거 (아파트 누리 -> 아파트누리)
    return t.lower()

def _match_pageid(md: dict, pid: str) -> bool:
    if not pid:
        return True
    pid_md = str((md or {}).get("page") or (md or {}).get("pageId") or "")
    src = str((md or {}).get("source") or "")
    url = str((md or {}).get("url") or "")
    return (pid_md == pid) or (f"pageId={pid}" in src) or (f"pageId={pid}" in url)

### [ADD] 로컬 업로드 소스 판별(경로 정규화 포함)
def _is_local_source(src: str) -> bool:
    s = _norm_source((src or "").replace("\\", "/"))
    up = (UPLOAD_DIR or "uploads").replace("\\", "/").strip("/")
    return (
        s.startswith("uploads/") or
        s.startswith(f"{up}/") or
        f"/{up}/" in f"/{s}" or
        (s.endswith(".pdf") and f"/{up}/" in f"/{s}")  # 옵션
    )

def _apply_local_bonus(hits: list[dict]):
    if not LOCAL_FIRST:
        return
    for h in hits:
        md = (h.get("metadata") or {})
        src = str(md.get("source") or md.get("url") or "")
        ### [FIX] 경로 표준화 기반의 로컬 판별 사용
        if _is_local_source(src):
            h["score"] = float(h.get("score") or 0.0) + LOCAL_BONUS

def _apply_page_hint(hits: List[dict], page_id: str | None):
    """pageId가 오면 '가산점'만 (soft). 잠그지 않음."""
    if not page_id or PAGE_FILTER_MODE.lower() != "soft":
        return
    for h in hits:
        if _match_pageid(h.get("metadata") or {}, page_id):
            h["score"] = float(h.get("score") or 0.0) + PAGE_HINT_BONUS

def _coalesce_single_source(hits: List[dict], q: str) -> List[dict]:
    """
    '최근/이슈/목록/정리' 같은 질의면 여러 출처가 섞이지 않게
    가장 점수가 높은 '단일 소스(=한 페이지)'만 남긴다.
    """
    if not SINGLE_SOURCE_COALESCE or not hits or not _COALESCE_TRIGGER.search(q or ""):
        return hits

    # source+page를 키로 클러스터링
    groups = {}
    for h in hits:
        md = h.get("metadata") or {}
        key = f"{md.get('url') or md.get('source') or ''}|p{md.get('page') or md.get('pageId') or ''}"
        groups.setdefault(key, []).append(h)

    scored = []
    for key, hs in groups.items():
        s = sum(float(x.get("score") or 0.0) for x in hs)
        scored.append((key, s, hs))
    if not scored:
        return hits

    scored.sort(key=lambda x: x[1], reverse=True)
    best_key, best_score, best_group = scored[0]
    total = sum(s for _, s, _ in scored) or 1.0
    if (best_score / total) >= COALESCE_THRESHOLD or len(scored) > 1:
        # 한 페이지로 수렴
        try:
            # _set_sticky(best_key.split("|p", 1)[0])  # 다음 질문 안정화
            if STICKY_FROM_COALESCE:
                try:
                    _set_sticky(best_key.split("|p", 1)[0])
                except Exception:
                    pass
        except Exception:
            pass
        return best_group
    return hits

SEARCH_LANGS = [s.strip() for s in os.getenv("SEARCH_LANGS", "ko,en").split(",") if s.strip()]

# 키워드 정제(Confluence CQL용)
_K_STOP = {"관련","내용","찾아줘","찾아","알려줘","정리","컨플루언스","에서","해줘",
           "무엇","어떤","대한","관련한","좀","좀만","계속","그리고","거나",
           "설명","소개","정의","토큰","token","코인", "해주세요","해 주세요","해줘요","해줘","주세요","대하여","대해","설명해줘","설명해주세요","말해줘","알려","알려줘","부탁해"
           }

_JOSA_RE = re.compile(
    r"(으로써|으로서|으로부터|라고는|라고도|라고|처럼|까지|부터|에게서|한테서|에게|한테|께|이며|이자|"
    r"으로|로서|로써|로부터|께서|와는|과는|에서는|에는|에서|에게|한테|와|과|을|를|은|는|이|가|의|에|도|만|랑|하고)$"
)

ACRONYM_RE = re.compile(r'^[A-Z]{2,5}$')

def _split_core_and_acronyms(tokens: List[str]) -> Tuple[List[str], List[str]]:
    core, acr = [], []
    for t in tokens:
        (acr if ACRONYM_RE.match(t) else core).append(t)
    return core, acr

# [ADD] ===== Sparse(키워드) 후보용 토크나이저/스코어러 =====

_TOK_RE = re.compile(r"[A-Za-z0-9가-힣]{2,}")

# 불용구/공손표현 꼬리 자르기 + 분절
_STOP_SUFFIX_RE = re.compile(
    r"(에\s*대하여|에\s*대해|에\s*대한|대하여|대해|"
    r"설명해\s*주세요|설명해주세요|설명해줘|알려줘|해줘요?|주세요)$"
)

def _preseg_stop_phrases(q: str) -> str:
    # 붙여쓰기일 때도 뭉친 꼬리를 띄우거나 제거
    return _STOP_SUFFIX_RE.sub(" ", q)

def _tokenize_query(q: str) -> List[str]:
    q = _preseg_stop_phrases(_basic_normalize(q))
    raw = _TOK_RE.findall(q or "")
    toks = []
    for t in raw:
        t = _STOP_SUFFIX_RE.sub("", t)  # 토큰 끝 꼬리 날리기
        t = _strip_josa(t)
        if t and t not in _K_STOP:
            toks.append(t)
    return list(dict.fromkeys(toks))[:12]

def _keyword_overlap_score(q_tokens: List[str], text: str, title: str = "") -> float:
    """간단한 스파스 점수: 토큰 교집합 비율 + 제목 매치 보너스"""
    if not q_tokens:
        return 0.0
    # 텍스트는 길 수 있으니 앞부분만 얇게 보지만, 제목은 풀로 본다
    body = (text or "")[:1200]
    hits = sum(1 for t in q_tokens if t in body)
    base = hits / max(4, len(q_tokens))          # 토큰 일부만 맞아도 0.x 점수
    if title:
        title_hits = sum(1 for t in q_tokens if t in title)
        if title_hits:
            base += TITLE_BONUS         # 제목 매치 보너스 (가산점)
    return float(base)

def _sparse_keyword_hits(q: str, limit: int = 150, space: Optional[str] = None) -> List[dict]:
    """
    docstore 전체를 훑어 '간단 키워드 매치' 기반 후보를 만든다.
    반환: pool_hits와 동일한 dict 목록({text, metadata, score})
    """
    if not q.strip():
        return []
    q_tokens = _tokenize_query(q)
    if not q_tokens:
        return []

    try:
        ds = getattr(vectorstore, "docstore", None)
        dct = getattr(ds, "_dict", {}) if ds else {}
    except Exception:
        return []

    out = []
    for d in dct.values():
        md = dict(d.metadata or {})
        src = str(md.get("source", ""))

        # (옵션) space 하드필터
        if space and SPACE_FILTER_MODE.lower() == "hard":
            s = (md.get("space") or "").strip()
            if not s or s.lower() != space.lower():
                continue

        title = (md.get("title") or "")
        score = _keyword_overlap_score(q_tokens, d.page_content or "", title)
        if score <= 0.0:
            continue

        # pool_hits 포맷으로 변환
        md["source"] = src
        out.append({
            "text": d.page_content or "",
            "metadata": md,
            "score": min(0.99, score),  # 스파스 자체 점수는 0~1 사이
        })

    # 상위 limit만
    out.sort(key=lambda h: h["score"], reverse=True)
    return out[:int(limit)]

# 1) 기본 정규화: 호환성 정규화 + 공백 축약
def _basic_normalize(text: str) -> str:
    # NFC로 정규화 (한글 결합형 안정화)
    t = unicodedata.normalize("NFC", text)
    # 전각/반각, 제어문자 정리 (필요하면 NFKC 고려)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# 2) 한국어 특화: 의미보존 범위에서 공백 제거(명사연속부)
#    - "지역 정보" → "지역정보" 같은 변이를 줄이기 위해 한글+숫자 연속부 내 공백 제거
_HANGUL_BLOCK = r"[가-힣0-9A-Za-z]"
def _collapse_korean_compounds(text: str) -> str:
    # 한글/숫자/영문 사이의 단일 공백 제거
    return re.sub(fr"(?<={_HANGUL_BLOCK})\s+(?={_HANGUL_BLOCK})", "", text)

# 3) 도메인 동의어/약어 매핑 (좌변 패턴 → 우변 표준형)
CANON_MAP = {
    r"\bNIA\b": "한국지능정보사회진흥원",
    r"한국\s*지능\s*정보\s*사회\s*진흥원": "한국지능정보사회진흥원",
    r"국가\s*정보화\s*진흥원": "한국지능정보사회진흥원",  # 옛 명칭
    r"지역\s*정보": "지역정보",
    r"아파트\s*누리": "아파트누리",
}

def _apply_canon_map(text: str) -> str:
    t = text
    for pat, rep in CANON_MAP.items():
        t = re.sub(pat, rep, t, flags=re.IGNORECASE)
    return t

def normalize_query(q: str) -> str:
    # 1단계: 기본 정규화
    t = _basic_normalize(q)
    # 2단계: 도메인 동의어 치환
    t = _apply_canon_map(t)
    # 3단계: 한국어 복합명사 공백 축약
    t = _collapse_korean_compounds(t)
    return t

def _apply_space_hint(hits: List[dict], space: Optional[str]):
    """SPACE_FILTER_MODE == soft 일 때, space 일치 항목에 가산점"""
    if not space or SPACE_FILTER_MODE.lower() != "soft":
        return
    sp = space.lower()
    for h in hits:
        s = ((h.get("metadata") or {}).get("space") or "").lower()
        if s and s == sp:
            h["score"] = float(h.get("score") or 0.0) + SPACE_HINT_BONUS

def _strip_josa(t: str) -> str:
    return _JOSA_RE.sub("", t)

def _to_mcp_keywords(q: str) -> str:
    toks = re.findall(r"[A-Za-z0-9가-힣]+", q)
    toks = [_strip_josa(t) for t in toks]                # ← 조사 제거
    toks = [t for t in toks if t and t not in _K_STOP and len(t) >= 2]
    hangs = [t for t in toks if re.search(r"[가-힣]", t)]
    keep = hangs if hangs else toks
    # 중복 제거 + 과도한 길이 방지
    dedup = list(dict.fromkeys(keep))
    return " ".join(dedup[:6])[:200] or q

logger = logging.getLogger("rag-proxy")
log = logging.getLogger(__name__)

# 전역 상태
retriever = None
vectorstore: Optional[FAISSStore] = None
drop_think_fn = None

INDEX_DIR = settings.INDEX_DIR
UPLOAD_DIR = settings.UPLOAD_DIR
CHUNK_SIZE = settings.CHUNK_SIZE
CHUNK_OVERLAP = settings.CHUNK_OVERLAP
EMBEDDING_MODEL = settings.EMBEDDING_MODEL
EMBEDDING_DEVICE = settings.EMBEDDING_DEVICE
DATA_CSV = settings.DATA_CSV

def ensure_dirs():
    Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)
    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

def _make_embedder() -> HuggingFaceEmbeddings:
    emb = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True}
    )
    # [ADD] e5 instruct 프리픽스 주입 (LangChain 메서드를 래핑)
    if settings.E5_USE_PREFIX and "e5" in EMBEDDING_MODEL.lower():
        _orig_eq = emb.embed_query
        _orig_ed = emb.embed_documents

        def _eq(text: str):
            return _orig_eq(f"query: {text}")

        def _ed(texts: List[str]):
            return _orig_ed([f"passage: {t}" for t in texts])

        emb.embed_query = _eq        # type: ignore
        emb.embed_documents = _ed    # type: ignore
    return emb

def _load_or_init_vectorstore() -> FAISSStore:
    try:
        faiss_file = Path(INDEX_DIR) / "index.faiss"
        pkl_file   = Path(INDEX_DIR) / "index.pkl"
        if faiss_file.exists() and pkl_file.exists():
            emb = _make_embedder()
            vs = FAISSStore.load_local(
                INDEX_DIR,
                embeddings=emb,
                allow_dangerous_deserialization=True
            )
            log.info("Loaded existing FAISS index from %s", INDEX_DIR)
            return vs
    except Exception as e:
        log.warning("load_local failed, starting empty: %s", e)
    # 없거나 로드 실패 → 빈 인덱스
    vs = _empty_faiss()
    log.info("Initialized empty FAISS index")
    return vs

def _empty_faiss() -> FAISSStore:
    emb = _make_embedder()
    dim = len(emb.embed_query("dim-probe"))
    index = faiss.IndexFlatL2(dim)   # normalize_embeddings=True이지만 기본 L2로 통일
    return FAISSStore(
        embedding_function=emb,
        index=index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={}
    )

def _reload_retriever():
    global retriever
    retriever = vectorstore.as_retriever(
        search_type=settings.SEARCH_TYPE,                       # [MOD]
        search_kwargs={"k": settings.RETRIEVER_K,               # [MOD]
                       "fetch_k": settings.RETRIEVER_FETCH_K}   # [MOD]
    )
    VS.retriever = retriever


# 인덱싱 동시성 방지 락
index_lock = asyncio.Lock()
# [ADD] MCP 폴백 동시 호출을 막는 락
mcp_lock = asyncio.Lock()

# 문서 고유 ID (중복 방지용)
def _doc_id(d: Document) -> str:
    src   = str(d.metadata.get("source", ""))
    page  = str(d.metadata.get("page", ""))
    kind  = str(d.metadata.get("kind", ""))
    chunk = str(d.metadata.get("chunk", ""))
    # 내용 기반 해시를 섞어 동일 page/kind라도 서로 다르게
    h = hashlib.sha1((d.page_content or "").encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{src}|p{page}|k{kind}|c{chunk}|h{h}"

def _upsert_docs_no_dup(docs: List[Document]) -> int:
    global vectorstore
    if not docs:
        return 0
    exist = set(getattr(vectorstore.docstore, "_dict", {}).keys())
    batch = set()  # ← 한 번의 add_documents 안에서의 중복을 걸러냄
    new_docs, ids = [], []
    for d in docs:
        did = _doc_id(d)
        if did in exist or did in batch:
            continue
        new_docs.append(d)
        ids.append(did)
        batch.add(did)
    if not new_docs:
        return 0
    vectorstore.add_documents(new_docs, ids=ids)
    vectorstore.save_local(INDEX_DIR)
    _reload_retriever()
    return len(new_docs)

def build_openai_messages(question: str, k: int = 5) -> Tuple[List[Dict[str, Any]], List[Document]]:
    global retriever, vectorstore

    # 1) 메타 질문이면 k 확장
    k_eff = 12 if META_PAT.search(question) else k

    # 2) 리트리브
    docs: List[Document] = []
    if retriever is not None:
        try:
            docs = retriever.invoke(question)
        except Exception:
            docs = []

    # 3) 질문에 파일명 힌트가 있으면 같은 파일 청크 우선
    m = re.search(r'([^\s"\'()]+\.pdf)', question, re.I)
    fname = m.group(1).lower() if m else None
    if fname and docs:
        filt = [d for d in docs if fname in str(d.metadata.get("source","")).lower()]
        if filt:
            docs = filt

    # 4) 제목/요약 청크를 docstore에서 찾아 맨 앞에 프리패스(있을 때)
    if fname and vectorstore is not None:
        try:
            ds = vectorstore.docstore._dict.values()
            title_doc = next((d for d in ds
                              if fname in str(d.metadata.get("source","")).lower()
                              and d.metadata.get("kind") == "title"), None)
            summary_doc = next((d for d in ds
                                if fname in str(d.metadata.get("source","")).lower()
                                and d.metadata.get("kind") == "summary"), None)
            prepend = []
            if title_doc and title_doc not in docs:
                prepend.append(title_doc)
            if summary_doc and summary_doc not in docs:
                prepend.append(summary_doc)
            if prepend:
                docs = prepend + docs
        except Exception:
            pass

    # 5) 제목/요약이 앞에 오도록 정렬(리트리브로 들어온 경우까지 정리)
    if docs:
        docs = sorted(docs, key=lambda d: 0 if d.metadata.get("kind") in ("title","summary") else 1)

    # 6) 컨텍스트 합치기
    context = "\n\n---\n\n".join(d.page_content for d in docs[:k_eff]) if docs else ""

    # 7) 시스템 프롬프트
    system = (
        "너의 사고과정은 절대로 출력하지 말고, 다음 컨텍스트로만 간결하고 정확하게 한국어로 답하라. "
        "모르면 정확히 다음 문장을 출력하라: 주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다"
    )
    if META_PAT.search(question):
        system += "\n\n요청이 '주제/개요/요약' 성격이면 핵심 내용을 2-3문장으로 요약하라."
    if context:
        system += f"\n\n컨텍스트:\n{context}"
    
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    return msgs, docs[:k_eff]

@app.on_event("startup")
def _startup():
    """데이터 CSV가 있으면 그걸로 인덱스 시작, 없으면 '빈 인덱스'로 시작(=PDF 바로 테스트 가능)."""
    global vectorstore, drop_think_fn
    ensure_dirs()
    try:
        # CSV가 있으면 기존 파이프라인으로 로딩
        if Path(DATA_CSV).exists():
            _rag_chain, _retr, vectorstore, drop_think_fn = build_rag_chain(
                data_path=DATA_CSV, index_dir=INDEX_DIR
            )
        else:
            # 빈 인덱스 (PDF 등 업로드 후 /update로 채우기)
            vectorstore = _load_or_init_vectorstore()
        _reload_retriever()
        logger.info("Startup complete. Index at %s", INDEX_DIR)
    except Exception as e:
        logger.exception("Startup failed: %s", e)
    VS.vectorstore = vectorstore
    VS.retriever = retriever


@app.get("/health")
def health():
    try:
        n = len(vectorstore.docstore._dict) if vectorstore else None
    except Exception:
        n = None
    return {"status": "ok" if vectorstore else "degraded",
            "doc_count": n, "index_dir": INDEX_DIR, "uploads": UPLOAD_DIR}

@app.get("/models")
async def models():
    return await proxy_get("/models")

@app.get("/v1/models")
def v1_models():
    import time
    return {
        "object": "list",
        "data": [
            {
                "id": "qwen3-30b-a3b-fp8-router",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "router"
            }
        ]
    }


from pydantic import BaseModel, Field

class AskPayload(BaseModel):
    """
    OpenAPI에 'q'가 반드시 필요한 필드로 표기되도록 하는 요청 스키마.
    - Open WebUI가 이 스키마를 읽고 q를 포함해 호출하게 됨
    - top_k / k, space / spaceKey 등도 받아서 query()에 그대로 전달
    """
    q: str = Field(..., description="사용자 질문 (필수)")
    k: int | None = Field(5, description="리트리브 상위 K", ge=1, le=50)
    top_k: int | None = Field(None, description="일부 클라이언트 호환용(내부에서 k로 매핑)")
    source: str | None = Field(None, description="소스 단일 필터")
    sources: list[str] | None = Field(None, description="소스 다중 필터")
    space: str | None = Field(None, description="Confluence space 힌트")
    spaceKey: str | None = Field(None, description="space와 동일(클라이언트 호환)")
    pageId: str | None = Field(None, description="Confluence pageId 힌트/잠금")
    messages: list[dict] | None = Field(None, description="대화형 호환 필드")
    sticky: bool | None = Field(None, description="이 요청에서 sticky 소스를 사용하지 않음")
    spaces: list[str] | None = Field(None, description="허용할 space 리스트(복수)")

    def to_query_dict(self) -> dict:
        d = self.model_dump(exclude_none=True)
        if "top_k" in d and "k" not in d:
            d["k"] = d.pop("top_k")
        if "spaceKey" in d and "space" not in d:
            d["space"] = d.pop("spaceKey")
        return d

@app.post(
    "/ask",
    summary="Ask",
    description="/ask → 내부적으로 /query 호출합니다. (벡터 검색 + 필요 시 폴백)",
)
async def ask(payload: AskPayload = Body(
    ...,
    example={"q": "NIA 사용자 시나리오", "k": 5}
)):
    """
    [중요] 여기를 Pydantic 모델로 바꿔야 OpenAPI에 '필수 q'가 반영되어
    Open WebUI가 빈 바디가 아닌 {'q': ...} 형태로 요청합니다.
    """
    # query()는 dict를 받으므로 모델을 dict로 변환
    return await query(payload.to_query_dict())

@app.post(
    "/qa",
    summary="Qa Compat",
    description="과거 호환용 엔드포인트. /ask와 동일하게 동작.",
)
async def qa_compat(payload: AskPayload = Body(
    ...,
    example={"q": "NIA 사용자 시나리오"}
)):
    """
    [중요] /qa도 동일 스키마를 쓰면 OpenAPI 상으로 동일한 요구사항이 노출됩니다.
    """
    return await query(payload.to_query_dict())


@app.post("/upload")
async def upload(file: UploadFile = File(...), overwrite: bool = Form(False)):
    ensure_dirs()
    safe_name = os.path.basename(file.filename).replace("\\", "/")
    dest = Path(UPLOAD_DIR) / safe_name
    if dest.exists() and not overwrite:
        raise HTTPException(409, f"File already exists: {dest}")
    try:
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        try:
            await file.close()
        except Exception:
            pass
    return {"saved": {"filename": safe_name, "path": str(dest), "bytes": dest.stat().st_size}}


@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    overwrite: bool = Form(False),
    parser: str = Form("auto"),
):
    global vectorstore, last_source   # ← 함수 제일 위에 둡니다

    if vectorstore is None:
        raise HTTPException(500, "vectorstore is not ready.")

    ensure_dirs()
    safe_name = os.path.basename(file.filename).replace("\\", "/")
    dest = Path(UPLOAD_DIR) / safe_name
    if dest.exists() and not overwrite:
        raise HTTPException(409, f"File exists: {dest}. Set overwrite=true")

    try:
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        await file.close()

    # 로딩/청킹
    try:
        docs = load_docs_any(
            dest,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            parser=parser,
        )
        if not docs:
            raise HTTPException(400, f"no docs parsed from {dest}")
    except ValueError as ve:
        raise HTTPException(415, str(ve))
    except Exception as e:
        raise HTTPException(500, f"parse failed: {e}")

    # 업서트 (중복 방지 + 락)
    try:
        async with index_lock:
            before = len(vectorstore.docstore._dict)
            added = _upsert_docs_no_dup(docs)
            after = len(vectorstore.docstore._dict)

            # ← 여기서 전역 업데이트 & sticky (전역선언은 함수 맨 위에서 이미 했음)
            last_source = _norm_source(str(dest))
            _set_sticky(last_source)

        return {
            "saved": {"path": str(dest), "bytes": dest.stat().st_size},
            "indexed": len(docs),
            "added": added,
            "doc_total": after,
            "parser": parser,
        }
    except Exception as e:
        raise HTTPException(500, f"index failed: {e}")


@app.post("/update")
async def update_index(payload: dict):
    global vectorstore, last_source   # ← 함수 제일 위에 둡니다

    if vectorstore is None:
        raise HTTPException(500, "vectorstore is not ready.")

    base = Path(UPLOAD_DIR).resolve()
    rel  = str((payload or {}).get("path") or "").replace("\\", "/")
    if not rel:
        raise HTTPException(400, "payload.path is required")

    candidate = (base / rel).resolve()
    try:
        candidate.relative_to(base)
    except Exception:
        raise HTTPException(403, "path must be inside uploads")

    if not candidate.exists():
        raise HTTPException(404, f"file not found: {candidate}")

    parser = (payload or {}).get("parser", "auto")

    # (주의) 여기서 last_source 건드리지 말 것!  ←←← 기존의 임시 sticky 라인 삭제

    # 1) 문서 로딩
    try:
        docs = load_docs_any(
            candidate,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            parser=parser,
        )
        if not docs:
            raise HTTPException(400, f"no docs parsed from {candidate}")
    except ValueError as ve:
        raise HTTPException(415, str(ve))
    except Exception as e:
        raise HTTPException(500, f"load failed: {e}")

    # 2) 업서트 + 저장 + 리트리버 갱신
    try:
        async with index_lock:
            before = len(vectorstore.docstore._dict)
            added = _upsert_docs_no_dup(docs)
            after = len(vectorstore.docstore._dict)

            # 업서트 성공 후에만 최근 소스/sticky 갱신
            last_source = _norm_source(str(candidate))
            _set_sticky(last_source)

        return {
            "ok": True,
            "indexed": len(docs),
            "added": added,
            "doc_total": after,
            "source": str(candidate),
            "chunks": CHUNK_SIZE,
            "overlap": CHUNK_OVERLAP,
            "parser": parser,
        }
    except Exception as e:
        log.exception("update failed")
        detail = str(e) or e.__class__.__name__
        raise HTTPException(500, f"update failed: {detail}")



# api/main.py 의 delete_index 전체를 아래로 교체

@app.delete("/delete")
async def delete_index(payload: Optional[dict] = None):
    """
    인덱스 삭제:
      - {"mode":"all"} → 인덱스 폴더 삭제 후 '빈 인덱스'로 재초기화
      - {"source":"uploads/파일.pdf"} → 해당 source만 제거
    """
    global vectorstore 
    global current_source, current_source_until, last_source 

    if vectorstore is None:
        raise HTTPException(500, "vectorstore is not ready.")

    payload = payload or {}
    mode = payload.get("mode")
    source = payload.get("source")

    if mode == "all":
        try:
            root = Path(INDEX_DIR)
            if root.exists():
                for child in root.iterdir():
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        try:
                            child.unlink()
                        except Exception:
                            pass

            # 빈 인덱스로 재초기화
            vectorstore = _empty_faiss()
            vectorstore.save_local(INDEX_DIR)
            _reload_retriever()
            VS.vectorstore = vectorstore
            VS.retriever  = retriever

            # sticky 상태도 초기화
            current_source = None
            current_source_until = 0.0
            last_source = None

            return {
                "deleted": "all",
                "doc_count": len(vectorstore.docstore._dict)
            }
        except Exception as e:
            raise HTTPException(500, f"delete all failed: {e}")


    if source:
        try:
            docstore = vectorstore.docstore._dict
            norm = _norm_source
            target = [doc_id for doc_id, doc in docstore.items()
                    if norm(str(doc.metadata.get("source",""))) == norm(source)]
            if not target:
                return {"deleted": 0, "reason": f"no documents with source={source}"}

            vectorstore.delete(target)
            vectorstore.save_local(INDEX_DIR)
            _reload_retriever()

            # ←←← (유지) 삭제한 소스가 sticky/last였다면 해제
            if _norm_source(source) == (current_source or ""):
                current_source = None
                current_source_until = 0.0
            if _norm_source(source) == (last_source or ""):
                last_source = None

            return {
                "deleted": len(target),
                "source": source,
                "doc_count": len(vectorstore.docstore._dict)
            }
        except Exception as e:
            raise HTTPException(500, f"delete by source failed: {e}")

    raise HTTPException(400, "Invalid payload. Use {'mode':'all'} or {'source':'uploads/파일.ext'}.")


@app.get("/index/stats")
def index_stats():
    """인덱스 요약(총 문서/벡터 수, 차원, 소스별/종류별 카운트)"""
    if vectorstore is None:
        raise HTTPException(500, "vectorstore is not ready.")
    try:
        ds = getattr(vectorstore, "docstore", None)
        dct = getattr(ds, "_dict", {}) if ds else {}
        total_docs = len(dct)

        by_source = Counter(str(d.metadata.get("source","")) for d in dct.values())
        by_kind   = Counter(str(d.metadata.get("kind","chunk")) for d in dct.values())

        # faiss index 통계
        idx = getattr(vectorstore, "index", None)
        dim = int(getattr(idx, "d", 0)) if idx is not None else None
        ntotal = int(getattr(idx, "ntotal", 0)) if idx is not None else None

        return {
            "doc_total": total_docs,
            "vector_total": ntotal,
            "dim": dim,
            "sources": [{"source": s or "(unknown)", "count": c} for s, c in by_source.most_common()],
            "kinds": [{"kind": k, "count": c} for k, c in by_kind.most_common()],
            "index_dir": INDEX_DIR,
            "uploads_dir": UPLOAD_DIR,
        }
    except Exception as e:
        raise HTTPException(500, f"stats failed: {e}")

@app.get("/index/sources")
def index_sources():
    """소스 파일(예: uploads/test.pdf)별 청크 수 목록"""
    if vectorstore is None:
        raise HTTPException(500, "vectorstore is not ready.")
    ds = vectorstore.docstore._dict
    by_source = Counter(str(d.metadata.get("source","")) for d in ds.values())
    items = [{"source": s or "(unknown)", "count": c} for s, c in by_source.most_common()]
    return {"total_sources": len(items), "items": items}

@app.get("/index/list")
def index_list(
    limit: int = 50,
    offset: int = 0,
    source: Optional[str] = None,
    kind: Optional[str] = None,
    contains: Optional[str] = None,
):
    """개별 청크 미리보기(페이지네이션/필터)"""
    if vectorstore is None:
        raise HTTPException(500, "vectorstore is not ready.")
    ds = vectorstore.docstore._dict

    def _norm_text(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").replace("\u00A0"," ").strip())
    norm_contains = _norm_text(contains) if contains else None

    rows = []
    for doc_id, doc in ds.items():
        src  = str(doc.metadata.get("source",""))
        knd  = doc.metadata.get("kind", "chunk")
        text = doc.page_content or ""
        title = str(doc.metadata.get("title",""))
        
        if source and _norm_source(src) != _norm_source(source):
            continue
        if kind and knd != kind:
            continue
        if norm_contains and (norm_contains not in _norm_text(text)) and (norm_contains not in _norm_text(title)):
            continue

        rows.append({
            "id": doc_id,
            "source": src,
            "kind": knd,
            "page": doc.metadata.get("page"),
            "chars": len(text),
            "preview": (text[:200] + ("…" if len(text) > 200 else "")),
        })

    total = len(rows)
    items = rows[offset: offset + limit]
    return {"total": total, "limit": limit, "offset": offset, "items": items}

@app.delete("/uploads/reset")
async def uploads_reset():
    try:
        root = Path(UPLOAD_DIR)
        if not root.exists():
            return {"status":"ok", "deleted":0}
        count = 0
        for child in root.iterdir():
            try:
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
                count += 1
            except Exception:
                pass
        return {"status":"ok", "deleted":count}
    except Exception as e:
        raise HTTPException(500, f"uploads reset failed: {e}")
    

@app.post("/query", include_in_schema=False)
async def query(payload: dict = Body(...)):
    global vectorstore, retriever, current_source, current_source_until
    fallback_attempted = False
    added = 0  # ←← 미리 초기화(하단에서 안전하게 notes에 넣거나 안 넣기 위함)

    # 1) 쿼리 추출
    q = (payload or {}).get("question") \
        or (payload or {}).get("query") \
        or (payload or {}).get("q") or ""
    if (not q.strip()) and isinstance((payload or {}).get("messages"), list):
        for m in reversed(payload["messages"]):
            if m.get("role") == "user" and m.get("content"):
                q = m["content"]; break
    if not q.strip():
        raise HTTPException(400, "question/query/q/messages is required")
    
    # [ADD] pageId 힌트만 추출(잠금 아님)
    page_id = (payload or {}).get("pageId")
    if not page_id:
        m_pid = re.search(r"pageId\s*=\s*(\d+)", q)
        if m_pid:
            page_id = m_pid.group(1)
    page_id = str(page_id) if page_id is not None else None

    if re.match(r"(?is)^\s*#{3}\s*task\s*:", q):  # ← 메타태스크는 RAG/MCP 건너뜀
        return {
            "hits": 0,
            "items": [],
            "contexts": [],
            "context_texts": [],
            "documents": [],
            "chunks": [],
            "notes": {"meta_task": True}
        }
        
    # === Direct-Answer 라우팅: 날짜/시간 등 상식형은 RAG/MCP를 건너뛰고 LLM이 바로 답 ===
    if _is_datetime_question(q):
        date_str, dow_str, time_str = _now_str_kst()
        sys = (
            "너는 RAG 컨텍스트 없이도 답할 수 있는 상식형 질문에 직접 답하는 어시스턴트다. "
            f"현재 표준시({TZ_NAME})는 {date_str} {dow_str}요일 {time_str} 이다. "
            "사용자가 날짜/요일/시간을 물으면 이 값을 활용해 간결히 한국어로 답하라. "
            "사고과정은 절대 출력하지 말라."
        )
        msgs = [
            {"role": "system", "content": sys},
            {"role": "user", "content": q},
        ]
        try:
            direct = await _call_llm(messages=msgs)
        except Exception:
            # LLM 호출 실패 시 서버 시계 기반으로 최소 응답 보장
            direct = f"오늘은 {date_str} {dow_str}요일입니다."

        return {
            "hits": 0,
            "items": [],
            "contexts": [],
            "context_texts": [],
            "documents": [],
            "chunks": [],
            "source_urls": [],
            "direct_answer": direct,
            "notes": {"routed": "no_rag", "reason": "datetime"}
        }

    if vectorstore is None:
        raise HTTPException(500, "vectorstore is not ready.")
    
    # 2) k / source
    k = (payload or {}).get("k") or (payload or {}).get("top_k") or 5
    try: k = int(k)
    except: k = 5
    src_filter = (payload or {}).get("source")
    src_list   = (payload or {}).get("sources")
    forced_page_id = _extract_page_id(src_filter)
    if not forced_page_id and page_id:
        forced_page_id = page_id
    src_set    = set(map(str, src_list)) if isinstance(src_list, list) and src_list else None

    # 요청 단위 sticky 비활성화 옵션
    sticky_flag = (payload or {}).get("sticky")
    ignore_sticky = (sticky_flag is False) or (isinstance(sticky_flag, str) and str(sticky_flag).lower() in ("0","false","no"))

    # sticky 적용 (유효기간 + 관련성 체크)
    now = time.time()
    if (not ignore_sticky) and not src_filter and not src_set and current_source and now < current_source_until:
        if (not STICKY_STRICT) or _sticky_is_relevant(q, current_source):
            src_filter = current_source
        else:
            current_source = None
            current_source_until = 0.0


    # "이 파일/첨부한 파일" 지시어면 최근 업로드 파일로 고정
    global last_source
    # 최근 업로드 파일 지시어(이 파일/첨부/해당 문서 등) → last_source 고정
    if not src_filter and not src_set and last_source and THIS_FILE_PAT.search(q):
        src_filter = last_source
        _set_sticky(last_source)  # 연속 질문 안정화


    # 3-A) 후보 선택은 MMR로(기존 그대로)
    try:
        docs = retriever.invoke(q)
    except Exception:
        docs = []

    # 빈 결과면 fallback (또는 예외 시에도)
    if not docs:
        try:
            docs = [d for d, _ in vectorstore.similarity_search_with_score(q, k=max(k*2, 10))]
        except Exception:
            docs = []
    # ... docs를 뽑은 직후, src 필터 적용 전에 추가
    # q_lc = q.lower()

    # 1) "xxx.pdf"가 질문에 포함되면 동일 소스만 우선
    m = re.search(r'([^\s"\'()]+\.pdf)', q, re.I)
    fname = m.group(1).lower() if m else None
    # >>> [ADD] 질문에 파일명이 있으면 동일 basename 소스로 바로 고정
    if fname:
        bn = Path(fname).name.lower()
        try:
            ds = getattr(vectorstore, "docstore", None)
            all_docs = getattr(ds, "_dict", {}) if ds else {}
            cands = [
                _norm_source(str(doc.metadata.get("source","")))
                for doc in all_docs.values()
                if Path(str(doc.metadata.get("source",""))).name.lower() == bn
            ]
            if cands:
                src_filter = cands[0]
                _set_sticky(src_filter)
        except Exception:
            pass

    # 2) 확장자 없이도 매칭(질문에 포함된 토큰이 소스 파일명에 들어가면)
    if not fname:
        def _canon_file(s: str) -> str:
            bn = Path(s).name.lower()
            for ext in ('.pdf', '.pptx', '.xlsx', '.txt', '.csv', '.md'):
                if bn.endswith(ext): bn = bn[:-len(ext)]
            return bn

        # 토큰 정규화: 조사/불용 접미 제거
        raw_tokens = re.findall(r'[\w\.\-\(\)가-힣]+', q.lower())
        norm_tokens = []
        for t in raw_tokens:
            if t.endswith("의"): t = t[:-1]
            # 필요한 경우 더 추가: t = t.replace("관련","").replace("에","")
            t = t.strip()
            if t: norm_tokens.append(t)

        # 후보 소스 basename 사전
        cand_sources = [str(d.metadata.get("source","")) for d in docs]
        bn_map = {_canon_file(s): s for s in cand_sources if s}

        hit_bn = None
        for t in norm_tokens:
            for bn in bn_map.keys():
                if t and t in bn:
                    hit_bn = bn; break
            if hit_bn: break

        if hit_bn:
            wanted = _norm_source(bn_map[hit_bn])
            docs = [d for d in docs if _norm_source(str(d.metadata.get("source",""))) == wanted]

    # [ADD] space 힌트 (Confluence space 키)
    space = (payload or {}).get("space") or (payload or {}).get("spaceKey")
    allowed_spaces = _resolve_allowed_spaces((payload or {}).get("spaces"))

    # 단일 space 힌트가 없다면, ENV나 클라의 allowed_spaces가 1개일 때 그걸 단일 힌트로 활용
    if not space and allowed_spaces and len(allowed_spaces) == 1:
        space = allowed_spaces[0]

    if isinstance(space, str):
        space = space.strip() or None

    # 3) 소스 필터 후 상위 k (4-C: 정규화 비교로 교체)
    if src_set:
        wanted = {_norm_source(str(s)) for s in src_set}
        docs = [d for d in docs if _norm_source(str((d.metadata or {}).get("source",""))) in wanted]
    elif src_filter:
        wanted = _norm_source(str(src_filter))
        docs = [d for d in docs if _norm_source(str((d.metadata or {}).get("source",""))) == wanted]

    if forced_page_id and PAGE_FILTER_MODE.lower() == "hard":
        def _doc_has_pid(d, pid):
            md = d.metadata or {}
            pid_md = str(md.get("page") or md.get("pageId") or "")
            url = str(md.get("url") or md.get("source") or "")
            return (pid_md == pid) or (f"pageId={pid}" in url)
        docs = [d for d in docs if _doc_has_pid(d, forced_page_id)]
    docs = docs[:k]

    # 3-B) 넉넉한 후보 풀을 구성하고 점수/메타를 붙인다
    def key_of(d):
        md = d.metadata or {}
        return (str(md.get("source","")), md.get("page"), md.get("kind","chunk"), md.get("chunk"))

    pool_hits = []   # ← rerank/pick_for_injection가 먹는 풀
    try:
        pairs = vectorstore.similarity_search_with_score(q, k=max(k*8, 80))
    except Exception:
        pairs = []

    # (선택) retriever 결과도 풀에 합치기
    base_docs = []
    try:
        base_docs = retriever.invoke(q) or []
    except Exception:
        base_docs = []

    # [ADD] --- 경량 스파스(키워드) 후보 융합 ---
    if ENABLE_SPARSE:
        sparse_hits = _sparse_keyword_hits(q, limit=SPARSE_LIMIT, space=space)
        _apply_space_hint(sparse_hits, space)
        _apply_page_hint(sparse_hits, page_id)   # ← sparse 후보들에 pageId 가산점
        pool_hits.extend(sparse_hits)

    # 유사도 점수 변환 (FAISS L2 → cos 유사도 근사)
    def _sim_from_dist(dist):
        try:
            s = 1.0 - float(dist)/2.0
            return max(0.0, min(1.0, s))
        except Exception:
            return 0.0
        
    # [DEBUG] pool 상위 몇 개 소스 찍기
    try:
        if pool_hits:
            _peek = []
            for h in pool_hits[:8]:
                md = (h.get("metadata") or {})
                _peek.append(md.get("source") or md.get("url") or "")
            log.info("pool peek sources: %s", _peek)
    except Exception:
        pass

    # 3-B-1) similarity_search_with_score 풀
    for d, dist in pairs:
        sim = _sim_from_dist(dist)
        md = dict(d.metadata or {})
        md["source"] = str(md.get("source",""))
        pool_hits.append({
            "text": d.page_content or "",
            "metadata": md,
            "score": sim,
        })

    # 3-B-2) retriever 풀(점수 없으면 0.5 기본 가중)
    for d in base_docs:
        md = dict(d.metadata or {})
        md["source"] = str(md.get("source",""))
        pool_hits.append({
            "text": d.page_content or "",
            "metadata": md,
            "score": 0.5,
        })

    # [ADD] dense 풀에도 space soft 보너스 적용
    _apply_space_hint(pool_hits, space)
    _apply_page_hint(pool_hits, page_id)
    _apply_local_bonus(pool_hits)
    # [OPTION] dense 후보에도 제목 매치 보너스(약하게)
    q_tokens = _tokenize_query(q)
    if q_tokens:
        for h in pool_hits:
            md = h.get("metadata") or {}
            title = (md.get("title") or "")
            if title and any(t in title for t in q_tokens):
                h["score"] = float(h.get("score") or 0.0) + (TITLE_BONUS * 0.5)

    # --- space 제한 적용 ---
    if allowed_spaces:
        mode = SPACE_FILTER_MODE.lower()
        allowed_set = {s.lower() for s in allowed_spaces}

        if mode == "hard":
            # 허용된 space만 남김
            pool_hits = [
                h for h in pool_hits
                if ((h.get("metadata") or {}).get("space","") or "").lower() in allowed_set
            ]
        else:
            # soft: 허용된 space에 가산점 부여
            for sp in allowed_spaces:
                _apply_space_hint(pool_hits, sp)

    # === 3-D) 의도 파악
    intent = parse_query_intent(q)

    m_art = _ARTICLE_RE.search(q)
    article_no = intent.get("article_no") if m_art else None
    m_ch = _CHAPTER_RE.search(q)
    chapter_no = int(m_ch.group(1)) if m_ch else None

    # 조문 질의여도, 사용자가 source를 명시한 경우에만 그 값으로 고정
    # (명시 안 했으면 sticky/last_source 그대로 유지)
    if intent.get("article_no") and ("source" in (payload or {})):
        src_filter = (payload or {}).get("source")

    # 여기에서 pool_hits에 최종 소스 필터 적용
    if src_set:
        wanted = {_norm_source(str(s)) for s in src_set}
        pool_hits = [h for h in pool_hits if _norm_source(h["metadata"].get("source","")) in wanted]
    elif src_filter:
        wanted = _norm_source(str(src_filter))
        pool_hits = [h for h in pool_hits if _norm_source(h["metadata"].get("source","")) == wanted]

    # --- 조문 텍스트 가산점/누락 플래그 ---
    def _bonus_for_article_text(h, artno: int) -> float:
        t = re.sub(r"[ \t\r\n│|¦┃┆┇┊┋丨ㅣ]", "", h.get("text") or "")
        return 0.15 if re.search(fr"제{artno}조", t) else 0.0

    # [추가] 장 보너스
    def _bonus_for_chapter_text(h, chapno: int) -> float:
        t = re.sub(r"[ \t\r\n│|¦┃┆┇┊┋丨ㅣ]", "", h.get("text") or "")
        return 0.20 if re.search(fr"제{chapno}장", t) else 0.0
    
    # --- 조문/장 보너스 적용
    if article_no:
        for h in pool_hits:
            h["score"] = float(h.get("score") or 0.0) + _bonus_for_article_text(h, article_no)

    if chapter_no:
        for h in pool_hits:
            h["score"] = float(h.get("score") or 0.0) + _bonus_for_chapter_text(h, chapter_no)
    
    # --- 누락 플래그는 '조'일 때만
    missing_article = False
    if article_no:
        have_meta = any((h.get("metadata") or {}).get("article_no") == article_no for h in pool_hits)
        have_text = any(_bonus_for_article_text(h, article_no) > 0 for h in pool_hits)
        missing_article = not (have_meta or have_text)

        # 조문 질의면 관련 히트에 가산점 부여
        # for h in pool_hits:
        #     h["score"] = float(h.get("score") or 0.0) + _bonus_for_article_text(h, article_no)

    if forced_page_id and PAGE_FILTER_MODE.lower() == "hard":
        def _hit_has_pid(h, pid):
            md  = h.get("metadata") or {}
            pid_md = str(md.get("page") or md.get("pageId") or "")
            url = str(md.get("url") or md.get("source") or "")
            return (pid_md == pid) or (f"pageId={pid}" in url)
        pool_hits = [h for h in pool_hits if _hit_has_pid(h, forced_page_id)]

    # 3-E) rerank + 의도기반 주입선택
    chosen = pick_for_injection(q, pool_hits, k_default=int(k) if isinstance(k, int) else 5)
    chosen = _coalesce_single_source(chosen, q)
    if intent.get("article_no") and src_filter and not any(
        (h.get("metadata") or {}).get("article_no") == intent["article_no"] for h in chosen
    ):
        try:
            pairs2 = vectorstore.similarity_search_with_score(q, k=160)
            pool_hits2 = []
            for d, dist in pairs2:
                if _norm_source(str((d.metadata or {}).get("source",""))) != _norm_source(str(src_filter)):
                    continue
                pool_hits2.append({
                    "text": d.page_content or "",
                    "metadata": dict(d.metadata, source=str(d.metadata.get("source",""))),
                    "score": _sim_from_dist(dist),
                })
            if pool_hits2:
                chosen = pick_for_injection(q, pool_hits2, k_default=int(k) if isinstance(k, int) else 5)
        except Exception:
            pass

    # 4) 응답 생성 (기존 포맷 유지)
    items, contexts = [], []
    for h in chosen:
        md = dict(h["metadata"])
        sc = float(h.get("score") or 0.0)
        entry = {
            "text": h["text"],
            "metadata": md,
            "score": round(sc, 4),
        }
        items.append(entry)
        contexts.append({
            "text": h["text"],
            "source": md.get("source"),
            "page": md.get("page"),
            "kind": md.get("kind", "chunk"),
            "score": round(sc, 4),
        })

    try:
        log.info(
            "Q=%r src_filter=%r sticky=%r page_id=%r pool=%d chosen=%d by_section=%s",
            q, src_filter, current_source, forced_page_id, len(pool_hits), len(items),
            dict(Counter((h.get('metadata') or {}).get('section_index', -1) for h in chosen))
        )
    except Exception:
        pass


    def _query_tokens(q: str) -> List[str]:
        raw = re.findall(r"[가-힣A-Za-z0-9]{2,}", q)
        toks = []
        for t in raw:
            t = _strip_josa(t)              # ← 추가: 조사 제거 (예: 아파트누리에 → 아파트누리)
            t = _apply_canon_map(t)         # ← 권장: 도메인 치환 일관화
            t = _collapse_korean_compounds(t)
            if t and t not in _K_STOP:
                toks.append(t)
        # 중복 제거
        return list(dict.fromkeys(toks))[:8]


    ctx_all = "\n".join(c["text"] for c in contexts)
    tokens = _query_tokens(q)
    core, acr = _split_core_and_acronyms(tokens)

    # 컨텍스트 본문에서 '핵심 토큰' 일부라도 맞는지
    core_hit = any(t in ctx_all for t in core)
    titles_meta = " ".join(
         f"{(h.get('metadata') or {}).get('title','')} {(h.get('metadata') or {}).get('source','')}"
         for h in items
     )
    
    acronym_hit = True if not acr else any(a in ctx_all or a in titles_meta for a in acr)
    anchors = _anchor_tokens_from_query(q)
    blob_norm = _norm_kr(ctx_all + " " + titles_meta)
    anchor_hit = (not anchors) or any(_norm_kr(a) in blob_norm for a in anchors)


    # NEED_FALLBACK = (
    #     (len(items) == 0) or
    #     (len(pool_hits) < max(10, k*2)) or
    #     missing_article or
    #     (acr and not acronym_hit) or
    #     (anchors and not anchor_hit)
    # )

    # tokens / acr / anchors / acronym_hit / anchor_hit 등이 계산된 상태
    anchor_miss = bool(anchors) and not anchor_hit
    acronym_miss = bool(acr) and not acronym_hit

    local_ok = _has_local_hits(items) or _has_local_hits(pool_hits)

    if local_ok:
        # 로컬 히트가 있어도 앵커/약어가 안 맞으면 폴백
        NEED_FALLBACK = (len(items) == 0) or missing_article or anchor_miss or acronym_miss
    else:
        NEED_FALLBACK = (
            (len(items) == 0) or
            (len(pool_hits) < max(10, k*2)) or
            missing_article or
            acronym_miss or
            anchor_miss
        )

    ### [ADD] '제 N장/조' 질의는 아이템이 이미 있으면 앵커 미스만으로 폴백 금지
    if (chapter_no or article_no) and items:
        NEED_FALLBACK = False

    # [PATCH] anchor_miss 단독이면 MCP 폴백 금지 (후보가 있는 경우)
    reasons = []
    if len(items) == 0: reasons.append("no_items")
    if len(pool_hits) < max(10, k*2): reasons.append("small_pool")
    if missing_article: reasons.append("missing_article")
    if (acr and not acronym_hit): reasons.append("acronym_miss")
    if (anchors and not anchor_hit): reasons.append("anchor_miss")
    log.info("fallback_check reasons=%s local_hits=%s", reasons, local_ok)

    # ▲ 위 reasons 계산 바로 아래에 추가
    if "anchor_miss" in reasons and reasons == ["anchor_miss"] and (len(items) > 0 or len(pool_hits) > 0):
        NEED_FALLBACK = False
        log.info("MCP skipped: anchor_miss only, but candidates exist (items=%d, pool=%d)", len(items), len(pool_hits))


    # [PATCH] '컨텍스트가 실제로 부족한' 이유일 때만 MCP 가동
    if NEED_FALLBACK and not DISABLE_INTERNAL_MCP and any(r in reasons for r in ("no_items", "small_pool", "missing_article")):
        try:
            fallback_attempted = True
            log.info("MCP fallback: calling Confluence MCP for query=%r", q)

            mcp_results = []
            spaces_for_mcp = allowed_spaces or ([space] if space else [None])
            for sp_hint in spaces_for_mcp:
                part = await mcp_search(q,  limit=5, timeout=20, space=sp_hint, langs=SEARCH_LANGS)
                mcp_results.extend(part or [])
                if mcp_results:  # 첫 space에서라도 결과가 나오면 멈춰도 OK (원하면 이어서 더 합쳐도 됨)
                    break

            if not mcp_results:
                q2 = _to_mcp_keywords(q)
                for sp_hint in spaces_for_mcp:
                    part = await mcp_search(q2, limit=5, timeout=20, space=sp_hint, langs=SEARCH_LANGS)
                    mcp_results.extend(part or [])
                    if mcp_results:
                        break

            if not mcp_results:
                ko = re.findall(r"[가-힣]{2,}", q)
                if ko:
                    best = _strip_josa(sorted(ko, key=len, reverse=True)[0])
                    for sp_hint in spaces_for_mcp:
                        part = await mcp_search(best, limit=5, timeout=20, space=sp_hint, langs=SEARCH_LANGS)
                        mcp_results.extend(part or [])
                        if mcp_results:
                            break


            # >>> [ADD] pageId가 강제되었으면 결과에서 해당 pageId만 남긴다.
            if forced_page_id and mcp_results:
                mcp_results = [r for r in mcp_results if _url_has_page_id(r.get("url"), forced_page_id) or str(r.get("id") or "") == forced_page_id]


            # === 결과 정규화/필터링 → items/contexts 로 변환 ===
            if mcp_results:
                items, contexts = [], []
                for r in mcp_results[:k]:
                    title = (r.get("title") or "").strip()
                    body  = (r.get("body") or r.get("excerpt") or r.get("text") or "")
                    body  = re.sub(r"@@@(?:hl|endhl)@@@", "", body)
                    text  = ((title + "\n\n") if title else "") + (body or "")

                    # 로그인 화면/빈 본문 제외
                    if not text.strip() or LOGIN_PAT.search(text):
                        continue

                    # pageId 보강
                    pid = (r.get("id") or "").strip()
                    if not pid:
                        m_pid = re.search(r"[?&]pageId=(\d+)", (r.get("url") or ""))
                        if m_pid:
                            pid = m_pid.group(1)

                    md = {
                        "source": r.get("url"),
                        "kind": "confluence",
                        "page": pid or None,
                        "space": r.get("space"),
                        "title": title,
                        "url": r.get("url"),
                    }

                    items.append({"text": text, "metadata": md, "score": 0.5})
                    contexts.append({
                        "text": text,
                        "source": md["source"],
                        "page": md["page"],
                        "kind": md["kind"],
                        "score": 0.5,
                    })

            if mcp_results:
                # 폴백 결과를 인덱스로 영구 업서트
                up_docs = []
                for r in mcp_results[:k]:
                    text_body = (r.get("body") or r.get("text") or r.get("excerpt") or "").strip()
                    if not text_body:
                        continue
                    src   = (r.get("url") or f"confluence:{r.get('id') or ''}").strip()
                    title = (r.get("title") or "").strip()
                    space = (r.get("space") or "").strip()

                    # [NEW] 제목 청크를 별도로 1개 추가 → build_openai_messages의 'title' 가중과도 맞물림
                    if title:
                        up_docs.append(Document(
                            page_content=f"[TITLE] {title}",
                            metadata={"source": src, "title": title, "space": space, "kind": "title"}
                        ))

                    text_full = ((title + "\n\n") if title else "") + text_body
                    for c in _chunk_text(text_full, CHUNK_SIZE, CHUNK_OVERLAP):
                        up_docs.append(Document(
                            page_content=c,
                            metadata={"source": src, "title": title, "space": space, "kind": "chunk"}
                        ))

                added = 0
                if up_docs:
                    async with index_lock:
                        added = _upsert_docs_no_dup(up_docs)
                        vectorstore.save_local(INDEX_DIR)
                        _reload_retriever()

                # 첫 결과를 최근 소스로 스티키 처리하면 연속 후속질문 안정적
                try:
                    if mcp_results and (mcp_results[0].get("url") or mcp_results[0].get("id")):
                        # _set_sticky(mcp_results[0].get("url") or f"confluence:{mcp_results[0].get('id')}")
                        want_sticky = False
                        if forced_page_id or THIS_FILE_PAT.search(q) or re.search(r'([^\s"\'()]+\.pdf)', q, re.I):
                            want_sticky = True
                        if STICKY_AFTER_MCP and want_sticky:
                            try:
                                first = mcp_results[0]
                                _set_sticky(first.get("url") or f"confluence:{first.get('id')}")
                            except Exception:
                                pass
                except Exception:
                    pass

        except Exception as e:
            log.error("MCP fallback failed: %s", "".join(traceback.format_exception(e)))


    # 장/조 질의면 필터 스킵
    if items and not (chapter_no or article_no):
        ctx_all_final = "\n".join(c["text"] for c in contexts)
        titles_meta_final = " ".join(
            f"{(h.get('metadata') or {}).get('title','')} {(h.get('metadata') or {}).get('source','')}"
            for h in items
        )
        core_final = [t for t in _query_tokens(q) if not ACRONYM_RE.match(t)]
        blob_norm_final = _norm_kr(ctx_all_final + " " + titles_meta_final)
        if core_final and not any(_norm_kr(t) in blob_norm_final for t in core_final):
            items, contexts = [], []

        if items:
            anchors = _anchor_tokens_from_query(q) + re.findall(r"[A-Z]{2,5}", q)
            if anchors and not any(_norm_kr(a) in blob_norm_final for a in anchors):
                items, contexts = [], []


        
        # 필터로 비어버리면 지금이라도 MCP 폴백 시도
        if not items and not DISABLE_INTERNAL_MCP:
            fallback_attempted = True
            mcp_results = []
            if forced_page_id:
                async with mcp_lock:
                    mcp_results = await mcp_search(f"id={forced_page_id}", limit=1, timeout=20,
                                                space=space, langs=SEARCH_LANGS)
            if not mcp_results:
                async with mcp_lock:
                    mcp_results = await mcp_search(q, limit=5, timeout=20,
                                                space=space, langs=SEARCH_LANGS)

            # 결과를 실제 items/contexts로 만들어주기 + 인덱스 반영
            if mcp_results:
                items, contexts, up_docs = _mcp_results_to_items(mcp_results, k=int(k) if isinstance(k, int) else 5)
                added = 0
                if up_docs:
                    async with index_lock:
                        added = _upsert_docs_no_dup(up_docs)
                        vectorstore.save_local(INDEX_DIR)
                        _reload_retriever()


    base_notes = {"missing_article": missing_article, "article_no": article_no}
    if fallback_attempted:
        # indexed/added는 MCP 업서트가 실제로 있었는지에 따라 값 세팅(위에서 added=0으로 시작)
        base_notes["fallback_used"] = True
        base_notes["indexed"] = (added > 0)
        if added > 0:
            base_notes["added"] = added

    if not items:
        return {
            "hits": 0,
            "items": [],
            "contexts": [],
            "context_texts": [],
            "documents": [],
            "chunks": [],
            "source_urls": [],
            "direct_answer": "주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다",
            "notes": base_notes | {"low_relevance": True},
        }

    # [OPT-ADD] items가 비면 contexts에서도 수집(방어 코팅)
    def _collect_source_urls_from_contexts(ctxs: list[dict]) -> list[str]:
        urls = []
        for c in ctxs or []:
            u = c.get("source") or c.get("url")
            if u and u not in urls:
                urls.append(u)
        return urls

    # (교체)
    src_urls = _collect_source_urls(items)
    if not src_urls:
        src_urls = _collect_source_urls_from_contexts(contexts)

    return {
        "hits": len(items),
        "items": items,
        "contexts": contexts,
        "context_texts": [it["text"] for it in items],
        "documents": items,
        "chunks": items,
        "source_urls": src_urls,  # ← 보강된 리스트
        "notes": base_notes,
    }
    # return {
    #     "hits": len(items),
    #     "items": items,
    #     "contexts": contexts,
    #     "context_texts": [it["text"] for it in items],
    #     "documents": items,
    #     "chunks": items,
    #     "source_urls": _collect_source_urls(items),
    #     "notes": base_notes,
    # }


# ------- helper: chunk text -------
def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    if not text:
        return []
    out = []
    i, n = 0, len(text)
    while i < n:
        j = min(n, i + int(size))
        out.append(text[i:j])
        if j == n:
            break
        i = j - int(overlap)
        if i < 0:
            i = 0
    return out


# [ADD] ------- helper: collect source urls -------
def _collect_source_urls(items: list[dict]) -> list[str]:
    """items[*].metadata.url or metadata.source에서 고유 URL만 추출"""
    urls: list[str] = []
    for it in items or []:
        md = (it.get("metadata") or {})
        url = md.get("url") or md.get("source")
        if url and url not in urls:
            urls.append(url)
    return urls


### [FIX] 로컬 히트 감지: source와 url 모두 확인 + 경로 정규화
def _has_local_hits(entries) -> bool:
    for it in entries or []:
        md = it.get("metadata") or {}
        src = str(md.get("source") or md.get("url") or "")
        if _is_local_source(src):
            return True
    return False

@app.post("/documents/upsert")
async def documents_upsert(payload: dict = Body(...)):
    """
    payload = {"docs":[{"id":"confluence:<page_id>",
                        "text":"<long text>",
                        "metadata":{"source":"confluence","title":"...","url":"...","space":"ENG"}}]}
    """
    global vectorstore
    if vectorstore is None:
        raise HTTPException(500, "vectorstore is not ready.")

    docs = (payload or {}).get("docs")
    if not isinstance(docs, list):
        raise HTTPException(400, "docs(list) is required")

    to_add = []
    for d in docs:
        text = (d.get("text") or "").strip()
        # meta = d.get("metadata") or {}
        meta_in = (d.get("metadata") or d.get("meta") or {})
        if not text:
            continue

        title = (meta_in.get("title") or "").strip()
        url   = (meta_in.get("url") or meta_in.get("source") or "").strip()
        space = (meta_in.get("space") or meta_in.get("spaceKey") or "").strip()
        source = url if url else str(d.get("id") or "confluence")

        doc_meta = dict(meta_in)               # ← 원본 메타 유지
        doc_meta.setdefault("source", source)
        doc_meta.setdefault("title", title)
        doc_meta.setdefault("space", space)
        doc_meta.setdefault("kind", "chunk")
        # [ADD] pageId가 오면 'page'에도 복사(중복 방지 ID 계산에 유리)
        if "page" not in doc_meta and doc_meta.get("pageId"):
            doc_meta["page"] = doc_meta.get("pageId")

        for c in _chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP):
            to_add.append(Document(
                page_content=c,
                metadata=doc_meta    # ← [CHANGE] 재구성하지 말고 보강된 메타 그대로 저장
            ))

    if not to_add:
        try:
            n = len(vectorstore.docstore._dict)
        except Exception:
            n = None
        return {"added": 0, "doc_total": n}

    try:
        async with index_lock:
            before = len(vectorstore.docstore._dict)
            added = _upsert_docs_no_dup(to_add)
            after  = len(vectorstore.docstore._dict)
        return {"added": added, "doc_total": after}
    except Exception as e:
        raise HTTPException(500, f"documents upsert failed: {e}")

# ← [ADD] 호환용 별칭: 브리지에서 /upsert, /v1/ingest, /v1/upsert 도 시도하므로 모두 수용
@app.post("/upsert")
async def upsert_alias(payload: dict = Body(...)):
    return await documents_upsert(payload)

@app.post("/v1/ingest")
async def v1_ingest_alias(payload: dict = Body(...)):
    return await documents_upsert(payload)

@app.post("/v1/upsert")
async def v1_upsert_alias(payload: dict = Body(...)):
    return await documents_upsert(payload)

app.include_router(smart_router)

from api.smart_router import ask_smart

@app.post("/v1/chat/completions")
async def v1_chat(payload: dict = Body(...)):
    import time as _time

    # 마지막 user 메시지 추출
    q = ""
    for m in payload.get("messages", []):
        if m.get("role") == "user" and m.get("content"):
            q = m["content"]
    if not q:
        q = payload.get("prompt") or ""
    if not q.strip():
        raise HTTPException(400, "user message required")

    # 1) 내부 /query 직접 호출
    page_id = payload.get("pageId") or payload.get("page_id")
    space   = payload.get("space") or payload.get("spaceKey")
    source  = payload.get("source")
    sources = payload.get("sources")
    spaces = payload.get("spaces")
    r = await query({
        "question": q,
        "pageId": page_id,
        "space": space,
        "source": source,
        "sources": sources,
        "spaces": spaces,  # ← 추가
        "sticky": False,  # 대화형 라우팅은 매질의 전환이 잦으므로 sticky 비활성
    })

    # 2) direct_answer가 있으면 그대로 사용
    content = r.get("direct_answer")

    # 3) 없으면 contexts를 컨텍스트로 하여 한 번 요약 생성
    if not content:
        ctx = "\n\n---\n\n".join(c.get("text", "") for c in r.get("contexts", [])[:6]).strip()
        if ctx:
            # 길이 가드 추가 (문자 기준 8천자 정도면 vLLM 쾌적)
            ctx = ctx[:8000]

            sys = (
                "사고과정을 출력하지 말고, 아래 컨텍스트에 **근거한 사실만** 한국어로 답하라. "
                "질문에 '제 N장' 또는 '제 N조'가 있으면, 컨텍스트에서 해당 장/조를 먼저 **그대로 인용(>)**하고, "
                "다음 줄에 한 문장 요약을 붙여라. "
                "컨텍스트에 없는 내용은 절대 추측하지 말고, 부족하면 '컨텍스트에 해당 정보가 없습니다'라고 말하라.\n\n"
                "컨텍스트:\n" + ctx
            )
            msgs = [
                {"role": "system", "content": sys},
                {"role": "user", "content": q},
            ]
            try:
                # 토큰/온도 명시(안정화)
                content = await _call_llm(messages=msgs, max_tokens=700, temperature=0.2)
            except Exception as e:
                log.warning("summarize failed: %s", e)
                content = "주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다"
        else:
            content = "주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다"

    allowed_list = r.get("source_urls", []) or []
    allowed_http = [u for u in allowed_list if isinstance(u, str) and u.startswith(("http://","https://"))]

    if content and allowed_http:
        def _keep_allowed(m):
            url = m.group(0)
            return url if any(url.startswith(a) for a in allowed_http) else ""
        content = re.sub(r'https?://[^\s\)\]]+', _keep_allowed, content)


    if allowed_list:
        src_block = "\n\n출처:\n" + "\n".join(f"- {u}" for u in allowed_list)
        content = (content or "") + src_block

    # 4) OpenAI 호환 스키마로 감싸서 반환
    return {
        "object": "chat.completion",
        "model": "qwen3-30b-a3b-fp8-router",
        "created": int(_time.time()),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "notes": r.get("notes", {}),
    }