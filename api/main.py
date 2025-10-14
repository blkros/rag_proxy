# rag-proxy/api/main.py
from __future__ import annotations
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
last_source: Optional[str] = None
# >>> [ADD] 최근 소스 잠금 상태 (연속 질문 안정화용)
current_source: Optional[str] = None
current_source_until: float = 0.0
STICKY_SECS = 180  # 최근 파일 기준 3분 잠금

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

SEARCH_LANGS = [s.strip() for s in os.getenv("SEARCH_LANGS", "ko,en").split(",") if s.strip()]

# 키워드 정제(Confluence CQL용)
_K_STOP = {"관련","내용","찾아줘","찾아","알려줘","정리","컨플루언스","에서","해줘",
           "무엇","어떤","대한","관련한","좀","좀만","계속","그리고","거나"}

_JOSA_RE = re.compile(
    r"(으로써|으로서|으로부터|라고는|라고도|라고|처럼|까지|부터|에게서|한테서|에게|한테|께|이며|이자|"
    r"으로|로서|로써|로부터|께서|와는|과는|에서는|에는|에서|에게|한테|와|과|을|를|은|는|이|가|의|에|도|만|랑|하고)$"
)

# [ADD] ===== Sparse(키워드) 후보용 토크나이저/스코어러 =====

_TOK_RE = re.compile(r"[A-Za-z0-9가-힣]{2,}")

def _tokenize_query(q: str) -> List[str]:
    raw = _TOK_RE.findall(q or "")
    toks = []
    for t in raw:
        t = _strip_josa(t)
        if t and t not in _K_STOP:
            toks.append(t)
    # 중복 제거, 너무 많은 토큰 방지
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
            base += settings.TITLE_BONUS         # 제목 매치 보너스 (가산점)
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
        if space and settings.SPACE_FILTER_MODE.lower() == "hard":
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

def _apply_space_hint(hits: List[dict], space: Optional[str]):
    """SPACE_FILTER_MODE == soft 일 때, space 일치 항목에 가산점"""
    if not space or settings.SPACE_FILTER_MODE.lower() != "soft":
        return
    sp = space.lower()
    for h in hits:
        s = ((h.get("metadata") or {}).get("space") or "").lower()
        if s and s == sp:
            h["score"] = float(h.get("score") or 0.0) + settings.SPACE_HINT_BONUS

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
                "id": "rag-proxy",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "rag-proxy"
            }
        ]
    }

# @app.post("/ask")
# async def ask(payload: dict):
#     """
#     /ask 도 /query 로 위임하여:
#       - 벡터 검색
#       - 필요 시 MCP(Confluence) 폴백
#     까지 한 번에 사용.
#     """
#     return await query(payload)

# @app.post("/qa")
# async def qa_compat(payload: dict = Body(...)):
#     return await query(payload)


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
    messages: list[dict] | None = Field(None, description="대화형 호환 필드")

    def to_query_dict(self) -> dict:
        d = self.model_dump(exclude_none=True)
        # 호환: top_k만 왔으면 k로 치환
        if "top_k" in d and "k" not in d:
            d["k"] = d.pop("top_k")
        # 호환: spaceKey만 왔으면 space로 치환
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
    # >>> FIX: 경로 탈출 방지 + 윈도우 역슬래시 제거
    safe_name = os.path.basename(file.filename).replace("\\", "/")
    dest = Path(UPLOAD_DIR) / safe_name
    if dest.exists() and not overwrite:
        raise HTTPException(409, f"File already exists: {dest}")
    try:
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        await file.close()
    return {"saved": {"filename": safe_name, "path": str(dest), "bytes": dest.stat().st_size}}  # ← safe_name 반영


@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    overwrite: bool = Form(False),
    parser: str = Form("auto"),
):
    """파일 저장 + 파싱/청킹 + 임베딩/업서트까지 원샷."""
    global vectorstore
    if vectorstore is None:
        raise HTTPException(500, "vectorstore is not ready.")

    ensure_dirs()
    # >>> FIX: 경로 탈출 방지
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
            global last_source
            last_source = _norm_source(str(dest))
            # >>> [ADD] 업로드 직후 최근 소스 잠금
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
    """
    업로드된 파일을 인덱스에 추가(멀티포맷: pdf/xlsx/pptx/txt, hwp 제외)
    payload 예: {"path": "uploads/문서.pdf", "parser": "auto|pdf_table"}
    """
    global vectorstore
    if vectorstore is None:
        raise HTTPException(500, "vectorstore is not ready.")

    rel = (payload or {}).get("path") or ""
    parser = (payload or {}).get("parser", "auto")
    if not rel:
        raise HTTPException(400, "path is required (e.g., 'uploads/doc.pdf')")

    # 경로 정규화
    path = Path(rel)
    if not path.is_absolute():
        path = Path(UPLOAD_DIR) / rel if not rel.startswith(UPLOAD_DIR) else Path(rel)
    if not path.exists():
        raise HTTPException(404, f"file not found: {path}")
    
    global last_source
    last_source = _norm_source(str(path))
    # >>> [ADD] 업데이트 직후 최근 소스 잠금
    _set_sticky(last_source)

    # 1) 문서 로딩
    try:
        docs = load_docs_any(
            path,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            parser=parser,           # ← 자동/표친화 감지 반영
        )
        if not docs:
            raise HTTPException(400, f"no docs parsed from {path}")
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
        return {
            "ok": True,
            "indexed": len(docs),
            "added": added,
            "doc_total": after,
            "source": str(path),
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

            # ←←← (수정) '전체 삭제'에서는 target 같은 부분 삭제 로직 쓰지 말고,
            #             인덱스를 통째로 재초기화합니다.
            vectorstore = _empty_faiss()
            vectorstore.save_local(INDEX_DIR)
            _reload_retriever()
            VS.vectorstore = vectorstore
            VS.retriever  = retriever

            # ←←← (추가) sticky/last 상태도 초기화
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

    rows = []
    for doc_id, doc in ds.items():
        src  = str(doc.metadata.get("source",""))
        knd  = doc.metadata.get("kind", "chunk")
        text = doc.page_content or ""

        if source and src != source:
            continue
        if kind and knd != kind:
            continue
        if contains and contains not in text:
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
    global vectorstore, retriever
    fallback_attempted = False 
    if vectorstore is None:
        raise HTTPException(500, "vectorstore is not ready.")

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

    # 2) k / source
    k = (payload or {}).get("k") or (payload or {}).get("top_k") or 5
    try: k = int(k)
    except: k = 5
    src_filter = (payload or {}).get("source")
    src_list   = (payload or {}).get("sources")
    src_set    = set(map(str, src_list)) if isinstance(src_list, list) and src_list else None

    # >>> [ADD] 스티키가 살아있으면 우선 적용
    now = time.time()
    if not src_filter and not src_set and current_source and now < current_source_until:
        src_filter = current_source

    # "이 파일/첨부한 파일" 지시어면 최근 업로드 파일로 고정
    global last_source
    if not src_filter and not src_set and last_source and THIS_FILE_PAT.search(q):
        src_filter = last_source
        # >>> [ADD] 지시어 등장 시 스티키 연장
        _set_sticky(last_source)

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
    if isinstance(space, str):
        space = space.strip() or None

    # 3) 소스 필터 후 상위 k (4-C: 정규화 비교로 교체)
    if src_set:
        wanted = {_norm_source(str(s)) for s in src_set}
        docs = [d for d in docs if _norm_source(str((d.metadata or {}).get("source",""))) in wanted]
    elif src_filter:
        wanted = _norm_source(str(src_filter))
        docs = [d for d in docs if _norm_source(str((d.metadata or {}).get("source",""))) == wanted]
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
    if settings.ENABLE_SPARSE:
        sparse_hits = _sparse_keyword_hits(q, limit=settings.SPARSE_LIMIT, space=space)
        # space soft 모드면 가산점 부여
        _apply_space_hint(sparse_hits, space)
        pool_hits.extend(sparse_hits)

    # 유사도 점수 변환 (FAISS L2 → cos 유사도 근사)
    def _sim_from_dist(dist):
        try:
            s = 1.0 - float(dist)/2.0
            return max(0.0, min(1.0, s))
        except Exception:
            return 0.0

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
    # [OPTION] dense 후보에도 제목 매치 보너스(약하게)
    q_tokens = _tokenize_query(q)
    if q_tokens:
        for h in pool_hits:
            md = h.get("metadata") or {}
            title = (md.get("title") or "")
            if title and any(t in title for t in q_tokens):
                h["score"] = float(h.get("score") or 0.0) + (settings.TITLE_BONUS * 0.5)

    # === 3-D) 의도 파악
    intent = parse_query_intent(q)

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
        # 텍스트 내 '제{n}조'가 OCR 노이즈(공백/세로바 등) 제거 후 보이면 가산점
        t = (h.get("text") or "")
        t2 = re.sub(r"[ \t\r\n│|¦┃┆┇┊┋丨ㅣ]", "", t)  # 공백/세로바 제거
        return 0.15 if re.search(fr"제{artno}조", t2) else 0.0

    missing_article = False
    if intent.get("article_no"):
        art = intent["article_no"]
        # 메타(article_no)로 잡혔거나, 텍스트 패턴으로라도 보이면 '존재'로 판단
        have_meta = any((h.get("metadata") or {}).get("article_no") == art for h in pool_hits)
        have_text = any(_bonus_for_article_text(h, art) > 0 for h in pool_hits)
        missing_article = not (have_meta or have_text)

        # 조문 질의면 관련 히트에 가산점 부여
        for h in pool_hits:
            h["score"] = float(h.get("score") or 0.0) + _bonus_for_article_text(h, art)

    # 3-E) rerank + 의도기반 주입선택
    chosen = pick_for_injection(q, pool_hits, k_default=int(k) if isinstance(k, int) else 5)
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
            "Q=%r src_filter=%r sticky=%r pool=%d chosen=%d by_section=%s",
            q, src_filter, current_source, len(pool_hits), len(items),
            dict(Counter((h.get('metadata') or {}).get('section_index', -1) for h in chosen))
        )
    except Exception:
        pass

    # === 컨텍스트에 질의 토큰이 하나도 없으면 폴백 플래그 ===
    def _query_tokens(q: str):
        toks = re.findall(r"[가-힣A-Za-z0-9]{2,}", q)
        toks = [t for t in toks if t not in _K_STOP]
        return toks[:8]

    ctx_all = "\n".join(c["text"] for c in contexts)
    has_any_query_token = any(t in ctx_all for t in _query_tokens(q))

    NEED_FALLBACK = (len(items) == 0) or (len(pool_hits) < max(10, k*2)) or missing_article or (not has_any_query_token)
    if NEED_FALLBACK and not DISABLE_INTERNAL_MCP:
        try:
            fallback_attempted = True
            log.info("MCP fallback: calling Confluence MCP for query=%r", q)
            async with mcp_lock:
                # [MOD] space/langs를 MCP로 전달
                mcp_results = await mcp_search(
                    q,
                    limit=5,
                    timeout=20,
                    space=space,              # [ADD]
                    langs=SEARCH_LANGS        # [ADD] ex) ["ko","en"]
                )


            # 2차: 키워드 정제
            if not mcp_results:
                q2 = _to_mcp_keywords(q)
                if q2 != q:
                    async with mcp_lock:
                        mcp_results = await mcp_search(q2, limit=5, timeout=20, space=space, langs=SEARCH_LANGS)

            # 3차: 가장 긴 한글 토큰
            if not mcp_results:
                ko = re.findall(r"[가-힣]{2,}", q)
                if ko:
                    best = sorted(ko, key=len, reverse=True)[0]
                    best = _strip_josa(best)
                    async with mcp_lock:
                        mcp_results = await mcp_search(best, limit=5, timeout=20, space=space, langs=SEARCH_LANGS)


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
                    text = (r.get("body") or r.get("text") or r.get("excerpt") or "").strip()
                    if not text:
                        continue
                    src   = (r.get("url") or f"confluence:{r.get('id') or ''}").strip()
                    title = (r.get("title") or "").strip()
                    space = (r.get("space") or "").strip()

                    for c in _chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP):
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
                        _set_sticky(mcp_results[0].get("url") or f"confluence:{mcp_results[0].get('id')}")
                except Exception:
                    pass

                return {
                    "hits": len(items),
                    "items": items,
                    "contexts": contexts,
                    "context_texts": [it["text"] for it in items],
                    "documents": items,
                    "chunks": items,
                    "notes": {"fallback_used": True, "indexed": True, "added": added}
                }
        except Exception as e:
            log.error("MCP fallback failed: %s", "".join(traceback.format_exception(e)))

    base_notes = {"missing_article": missing_article, "article_no": intent.get("article_no")}
    if fallback_attempted:
        base_notes["fallback_used"] = True
        base_notes["indexed"] = False

    return {
        "hits": len(items),
        "items": items,
        "contexts": contexts,
        "context_texts": [it["text"] for it in items],
        "documents": items,
        "chunks": items,
        "notes": base_notes
    }



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