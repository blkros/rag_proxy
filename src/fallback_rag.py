# rag-proxy/src/fallback_rag.py
from __future__ import annotations

import hashlib
from typing import List, Tuple

import asyncio
from src.ext.confluence_mcp import mcp_search
from src.rag_pipeline import Document
from src.vectorstore import vectorstore  # similarity_search_with_score, add_documents / upsert 등
from src.llm_pipeline import chat_with_context

# 튜닝 파라미터(필요하면 조절)
TOP_K = 8
MIN_HITS = 3
SCORE_TH = 0.40
MCP_LIMIT = 6
CHUNK_SZ = 1400
CH_OVER = 180


def _chunk(text: str) -> List[str]:
    if not text:
        return []
    out: List[str] = []
    i, n = 0, len(text)
    while i < n:
        j = min(n, i + CHUNK_SZ)
        out.append(text[i:j])
        if j == n:
            break
        i = j - CH_OVER
    return out


def _doc_key(uri: str, version, chunk_text: str) -> str:
    digest = hashlib.sha1(chunk_text.encode("utf-8")).hexdigest()[:16]
    return f"{uri}::v{version or '0'}::{digest}"


def _good(hits: List[Tuple[Document, float]]) -> bool:
    if len(hits) >= MIN_HITS:
        return True
    return bool(hits and hits[0][1] >= SCORE_TH)


def _retrieve_local(query: str) -> List[Tuple[Document, float]]:
    return vectorstore.similarity_search_with_score(query, k=TOP_K)


def _upsert_chunks(uri: str, title: str, space: str, version, chunks: List[str]) -> int:
    new_docs: List[Document] = []
    for c in chunks:
        key = _doc_key(uri, version, c)
        # 중복방지: vectorstore에 contains가 있으면 활용
        if hasattr(vectorstore, "contains") and vectorstore.contains(key):
            continue
        meta = {
            "id": key,
            "source": uri,
            "type": "confluence",
            "space": space,
            "title": title,
            "version": version,
            "origin": "mcp-live",
        }
        new_docs.append(Document(page_content=c, metadata=meta))

    if not new_docs:
        return 0

    # 네 파이프라인의 임베딩/저장을 그대로 사용
    if hasattr(vectorstore, "add_documents"):
        vectorstore.add_documents(new_docs)  # 내부 임베딩 + 영속화
    elif hasattr(vectorstore, "upsert"):
        vectorstore.upsert(new_docs)  # 구현에 따라 내부 임베딩 포함일 수 있음
    else:
        raise RuntimeError("vectorstore에 add_documents 또는 upsert가 필요합니다.")

    return len(new_docs)


def answer_with_fallback(question: str, space: str | None = None) -> dict:
    # 1) 로컬 검색
    hits = _retrieve_local(question)

    # 2) 부족하면 MCP로 보충 → 영속 업서트 → 재검색
    if not _good(hits):
        mcp_results = asyncio.run(mcp_search(question, limit=MCP_LIMIT))
        for h in mcp_results:
            url   = (h.get("url")   or h.get("link") or h.get("webui") or "").strip()
            title = (h.get("title") or h.get("name") or "").strip()
            txt   = (h.get("body")  or h.get("content") or h.get("excerpt") or "").strip()
            if not (url and txt):
                continue

            chunks = _chunk(txt)

            _upsert_chunks(
                uri=url,
                title=title,
                space=space or "",
                version=None,
                chunks=chunks,
            )

        hits = _retrieve_local(question)

    # 3) 컨텍스트 구성 + LLM 호출
    ctx_docs = [d for (d, _) in hits[:TOP_K]]
    contexts: List[str] = []
    for d in ctx_docs:
        md = d.metadata or {}
        head = f"[{md.get('space','')}/{md.get('title','')}] {md.get('source','')}"
        contexts.append(f"{head}\n{d.page_content}")

    system = "Answer strictly from the provided context. If not contained, say you don't know."
    answer = chat_with_context(question=question, contexts=contexts, system_prompt=system)

    sources = [
        {
            "source": (d.metadata or {}).get("source"),
            "title": (d.metadata or {}).get("title"),
            "space": (d.metadata or {}).get("space"),
            "version": (d.metadata or {}).get("version"),
        }
        for d in ctx_docs
    ]
    return {"answer": answer, "sources": sources}
