# api/main.py
from __future__ import annotations
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from typing import List, Dict, Any, Optional
from pathlib import Path
import shutil, os, logging

from src.utils import proxy_get, call_chat_completions, drop_think
from src.rag_pipeline import build_rag_chain, Document  # 파이프라인은 그대로 둠
from src.loaders import load_docs_any

# empty FAISS 빌드를 위한 보조들
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS as FAISSStore
from langchain_community.docstore.in_memory import InMemoryDocstore

app = FastAPI()
logger = logging.getLogger("rag-proxy")

# 전역 상태
retriever = None
vectorstore: Optional[FAISSStore] = None
drop_think_fn = None

INDEX_DIR = os.getenv("INDEX_DIR", "faiss_index")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
DATA_CSV = os.getenv("DATA_CSV", "data/민생.csv")  # 없으면 빈 인덱스로 시작
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

def ensure_dirs():
    Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)
    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

def _make_embedder() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True}
    )

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

def build_openai_messages(question: str, k: int = 5) -> List[Dict[str, Any]]:
    context = ""
    if retriever is not None:
        try:
            docs: List[Document] = (
                retriever.get_relevant_documents(question)
                if hasattr(retriever, "get_relevant_documents")
                else retriever.invoke(question)
            )
        except Exception:
            docs = []
        if docs:
            context = "\n\n---\n\n".join(d.page_content for d in docs[:k])

    system = (
        "너의 사고과정은 절대로 출력하지 말고, 다음 컨텍스트로만 간결하고 정확하게 한국어로 답하라. "
        "모르면 정확히 다음 문장을 출력하라: 주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다"
    )
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
            vectorstore = _empty_faiss()
            vectorstore.save_local(INDEX_DIR)
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

@app.post("/update")
async def update_index(payload: dict):
    """
    업로드된 파일을 인덱스에 추가(멀티포맷: pdf/xlsx/pptx/txt, hwp 제외)
    payload 예: {"path": "uploads/문서.pdf"}
    """
    global vectorstore
    if vectorstore is None:
        raise HTTPException(500, "vectorstore is not ready.")

    rel = (payload or {}).get("path") or ""
    if not rel:
        raise HTTPException(400, "path is required (e.g., 'uploads/doc.pdf')")
    path = Path(rel)
    if not path.is_absolute():
        path = Path(UPLOAD_DIR) / rel if not rel.startswith(UPLOAD_DIR) else Path(rel)
    if not path.exists():
        raise HTTPException(404, f"file not found: {path}")

    try:
        docs = load_docs_any(path, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    except ValueError as ve:
        raise HTTPException(415, str(ve))
    except Exception as e:
        raise HTTPException(500, f"load failed: {e}")

    try:
        before = len(vectorstore.docstore._dict)
        vectorstore.add_documents(docs)
        vectorstore.save_local(INDEX_DIR)
        _reload_retriever()
        after = len(vectorstore.docstore._dict)
        return {"indexed": len(docs), "added": after - before, "doc_total": after,
                "source": str(path), "chunks": CHUNK_SIZE, "overlap": CHUNK_OVERLAP}
    except Exception as e:
        raise HTTPException(500, f"update failed: {e}")

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
            if Path(INDEX_DIR).exists():
                shutil.rmtree(INDEX_DIR)
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