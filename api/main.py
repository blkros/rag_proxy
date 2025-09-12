# api/main.py
from __future__ import annotations
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from typing import List, Dict, Any, Optional
from pathlib import Path
import shutil, os, logging, re
from fastapi.responses import RedirectResponse

from src.utils import proxy_get, call_chat_completions, drop_think
from src.rag_pipeline import build_rag_chain, Document
from src.loaders import load_docs_any
from fastapi.middleware.cors import CORSMiddleware
from src.config import settings
import asyncio, hashlib
from collections import Counter

# empty FAISS 빌드를 위한 보조들
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS as FAISSStore
from langchain_community.docstore.in_memory import InMemoryDocstore


META_PAT  = re.compile(r"(주제|개요|요약|무엇|무슨\s*내용)", re.I)
TITLE_PAT = re.compile(r"(제목|title|문서명|파일명)", re.I)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
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
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True}
    )

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
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 40})

# 인덱싱 동시성 방지 락
index_lock = asyncio.Lock()

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

def build_openai_messages(question: str, k: int = 5) -> List[Dict[str, Any]]:
    global retriever, vectorstore

    # 1) 메타 질문이면 k 확장
    k_eff = 12 if META_PAT.search(question) else k

    # 2) 리트리브
    docs: List[Document] = []
    if retriever is not None:
        try:
            docs = (retriever.get_relevant_documents(question)
                    if hasattr(retriever, "get_relevant_documents")
                    else retriever.invoke(question))
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

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

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

@app.post("/ask")
async def ask(payload: dict):
    q = (payload or {}).get("question") or ""
    if not q.strip():
        raise HTTPException(400, "question is required")
    msgs = build_openai_messages(q, k=5)
    res = await call_chat_completions(messages=msgs, temperature=0)
    try:
        content = res["choices"][0]["message"]["content"]
    except Exception:
        content = str(res)
    return {"answer": drop_think(content)}

@app.post("/upload")
async def upload(file: UploadFile = File(...), overwrite: bool = Form(False)):
    ensure_dirs()
    dest = Path(UPLOAD_DIR) / file.filename
    if dest.exists() and not overwrite:
        raise HTTPException(409, f"File already exists: {dest}")
    try:
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        await file.close()
    return {"saved": {"filename": file.filename, "path": str(dest), "bytes": dest.stat().st_size}}

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
    dest = Path(UPLOAD_DIR) / file.filename
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

@app.delete("/delete")
async def delete_index(payload: Optional[dict] = None):
    """
    인덱스 삭제:
      - {"mode":"all"} → 인덱스 폴더 삭제 후 '빈 인덱스'로 재초기화
      - {"source":"uploads/파일.pdf"} → 해당 source만 제거
    """
    global vectorstore
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
            vectorstore = _empty_faiss()
            vectorstore.save_local(INDEX_DIR)
            _reload_retriever()
            return {"deleted": "all", "doc_count": len(vectorstore.docstore._dict)}
        except Exception as e:
            raise HTTPException(500, f"delete all failed: {e}")

    if source:
        try:
            docstore = vectorstore.docstore._dict
            target = [doc_id for doc_id, doc in docstore.items()
                      if str(doc.metadata.get("source", "")) == str(source)]
            if not target:
                return {"deleted": 0, "reason": f"no documents with source={source}"}
            vectorstore.delete(target)
            vectorstore.save_local(INDEX_DIR)
            _reload_retriever()
            return {"deleted": len(target), "source": source, "doc_count": len(vectorstore.docstore._dict)}
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
async def query_compat(payload: dict):
    """
    Open WebUI가 호출하는 /query 바디를 /ask 형태로 정규화해 처리.
    지원 키: question | query | q | messages (OpenAI 포맷)
    선택 파라미터: k | top_k
    """
    q = (payload or {}).get("question") \
        or (payload or {}).get("query") \
        or (payload or {}).get("q") \
        or ""

    # messages 포맷(최근 user 메시지) 지원
    if (not q.strip()) and isinstance((payload or {}).get("messages"), list):
        msgs = payload["messages"]
        for m in reversed(msgs):
            if m.get("role") == "user" and m.get("content"):
                q = m["content"]
                break

    if not q.strip():
        raise HTTPException(400, "question/query/q/messages is required")

    k = (payload or {}).get("k") or (payload or {}).get("top_k") or 5
    try:
        k = int(k)
    except Exception:
        k = 5

    msgs = build_openai_messages(q, k=k)
    res = await call_chat_completions(messages=msgs, temperature=0)
    try:
        content = res["choices"][0]["message"]["content"]
    except Exception:
        content = str(res)
    return {"answer": drop_think(content)}