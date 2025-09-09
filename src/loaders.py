# src/loaders.py
from __future__ import annotations
from pathlib import Path
from typing import List, Union
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os, csv
import chardet
from pypdf import PdfReader
from pptx import Presentation
import openpyxl
from src.config import settings

# -----------------------------
# 공통: 청킹
# -----------------------------
def _chunk(docs: List[Document], chunk_size=800, chunk_overlap=120) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    out: List[Document] = []
    for d in docs:
        parts = splitter.split_text(d.page_content or "")
        for i, p in enumerate(parts):
            md = dict(d.metadata)
            md["chunk"] = i
            out.append(Document(page_content=p, metadata=md))
    return out

# -----------------------------
# TXT
# -----------------------------
def load_txt(path: Path) -> List[Document]:
    raw = path.read_bytes()
    enc = chardet.detect(raw).get("encoding") or "utf-8"
    text = raw.decode(enc, errors="ignore")
    return [Document(page_content=text, metadata={"source": str(path), "type": "txt"})]

# -----------------------------
# PDF (기본 텍스트 파서)
# -----------------------------
def load_pdf(path: Path) -> List[Document]:
    reader = PdfReader(str(path))
    docs: List[Document] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(Document(page_content=text, metadata={
                "source": str(path), "type": "pdf", "page": i
            }))
    return docs

# -----------------------------
# PPTX
# -----------------------------
def load_pptx(path: Path) -> List[Document]:
    prs = Presentation(str(path))
    docs: List[Document] = []
    for i, slide in enumerate(prs.slides, start=1):
        lines = []
        for shp in slide.shapes:
            if getattr(shp, "has_text_frame", False) and shp.text_frame:
                for p in shp.text_frame.paragraphs:
                    if p.runs:
                        lines.append("".join(r.text or "" for r in p.runs))
                    else:
                        lines.append(p.text or "")
            if getattr(shp, "has_table", False):
                tbl = shp.table
                for row in tbl.rows:
                    cells = [(c.text or "").strip() for c in row.cells]
                    lines.append("\t".join(cells))
        text = "\n".join([ln for ln in lines if ln.strip()])
        if text.strip():
            docs.append(Document(page_content=text, metadata={
                "source": str(path), "type": "pptx", "slide": i
            }))
    return docs

# -----------------------------
# XLSX
# -----------------------------
def load_xlsx(path: Path) -> List[Document]:
    wb = openpyxl.load_workbook(str(path), data_only=True, read_only=True)
    docs: List[Document] = []
    for ws in wb.worksheets:
        lines = []
        for row in ws.iter_rows(values_only=True):
            line = "\t".join("" if v is None else str(v) for v in row)
            if line.strip():
                lines.append(line)
        text = "\n".join(lines)
        if text.strip():
            docs.append(Document(page_content=text, metadata={
                "source": str(path), "type": "xlsx", "sheet": ws.title
            }))
    return docs

# -----------------------------
# CSV (제목/요약 메타 청크 추가)
# -----------------------------
def load_csv(path: Path) -> List[Document]:
    raw = path.read_bytes()
    enc = chardet.detect(raw).get("encoding") or "utf-8"
    text = raw.decode(enc, errors="ignore")

    docs: List[Document] = []
    rows_norm: list[dict] = []
    lines: list[str] = []

    # DictReader 우선 (헤더가 있을 때)
    rdr = csv.DictReader(text.splitlines())
    if rdr.fieldnames:
        for r in rdr:
            r = { (k or "").strip(): (v or "").strip() for k, v in r.items() }
            rows_norm.append(r)
            k = r.get("항목") or r.get("항목명") or r.get("key") or ""
            v = r.get("내용") or r.get("세부내용") or r.get("value") or ""
            line = f"{k}: {v}" if k else v
            if line.strip():
                lines.append(line)
    else:
        # 헤더가 없으면 일반 reader로 라인 합치기
        rdr2 = csv.reader(text.splitlines())
        for r in rdr2:
            line = ",".join("" if v is None else str(v) for v in r).strip()
            if line:
                lines.append(line)

    if lines:
        docs.append(Document(
            page_content="\n".join(lines),
            metadata={"source": str(path), "type": "csv"}
        ))

    # === TITLE ===
    title = None
    for key in ("제목", "주제", "문서명", "title", "subject"):
        for r in rows_norm:
            if r.get(key):
                title = r[key]
                break
        if title:
            break
    if not title:
        # 항목/항목명에 적절한 값이 있으면 사용
        for r in rows_norm:
            maybe = r.get("항목") or r.get("항목명")
            if maybe and len(maybe) >= 2:
                title = maybe
                break
    if not title:
        title = path.stem

    docs.append(Document(
        page_content=f"[TITLE] {title}",
        metadata={"source": str(path), "kind": "title"}
    ))

    # === SUMMARY === (상위 몇 개를 key=value로)
    pairs = []
    for r in rows_norm[:6]:
        k = (r.get("항목") or r.get("항목명") or "").strip()
        v = (r.get("내용") or r.get("세부내용") or "").strip()
        if k and v:
            pairs.append(f"{k}={v}")
    if pairs:
        docs.append(Document(
            page_content=f"[SUMMARY] {path.name} 개요: " + "; ".join(pairs),
            metadata={"source": str(path), "kind": "summary"}
        ))
    return docs

# -----------------------------
# PDF (표 친화 자동 로더)
# -----------------------------
def _norm(s: str | None) -> str:
    return " ".join((s or "").split())

def _table_to_sentences(table: list[list[str]]) -> list[str]:
    rows = [[_norm(c) for c in r] for r in table if any(_norm(c) for c in r)]
    if not rows:
        return []
    headers = rows[0]
    out = []
    for r in rows[1:]:
        pairs = []
        for h, v in zip(headers, r):
            if _norm(h) and _norm(v):
                pairs.append(f"{h}={v}")
        if pairs:
            out.append(", ".join(pairs))
    return out

def _dedup(docs: List[Document]) -> List[Document]:
    seen = set()
    out: List[Document] = []
    for d in docs:
        key = d.page_content.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(d)
    return out

def _extract_pdf_title(path: Path) -> str | None:
    # 1) PDF 메타데이터 title
    try:
        meta = PdfReader(str(path)).metadata or {}
        t = None
        if hasattr(meta, "title"):
            t = meta.title
        elif isinstance(meta, dict):
            t = meta.get("/Title") or meta.get("Title")
        if t and str(t).strip():
            return str(t).strip()
    except Exception:
        pass
    # 2) 첫 페이지 첫 줄 휴리스틱 (pdfplumber 있으면)
    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            if pdf.pages:
                text = pdf.pages[0].extract_text() or ""
                for ln in (l.strip() for l in text.splitlines()):
                    if len(ln) >= 3:
                        return ln[:120]
    except Exception:
        pass
    # 3) 폴백: 파일명
    return path.stem

def _load_pdf_auto(path: Path) -> List[Document]:
    """표→문장 + 텍스트 스니펫을 함께 만들고,
       표 문장이 적으면 기본 파서 결과와 병합."""
    try:
        import pdfplumber
    except Exception:
        return load_pdf(path)

    docs: List[Document] = []
    first_text_lines: list[str] = []
    table_sentence_cnt = 0

    with pdfplumber.open(str(path)) as pdf:
        for pi, page in enumerate(pdf.pages, start=1):
            # 1) 표 → 문장
            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []
            for t in tables:
                sents = _table_to_sentences(t)
                table_sentence_cnt += len(sents)
                for s in sents:
                    docs.append(Document(
                        page_content=f"[SOURCE:{path.name}] {s}",
                        metadata={"source": str(path), "page": pi, "kind": "table_row"},
                    ))
            # 2) 텍스트 스니펫
            text = page.extract_text() or ""
            lines = [_norm(l) for l in text.splitlines() if _norm(l)]
            if pi == 1:
                first_text_lines = lines[:10]
            if lines:
                snippet = " ".join(lines[:20])[:1200]
                if snippet:
                    docs.append(Document(
                        page_content=f"[SOURCE:{path.name}] {snippet}",
                        metadata={"source": str(path), "page": pi, "kind": "page_text"},
                    ))

    # 3) 표가 거의 없으면 기본 파서와 병합
    threshold = settings.PDF_TABLE_THRESHOLD
    if table_sentence_cnt < threshold:
        legacy = load_pdf(path)
        docs = _dedup(legacy + docs)

    # 4) 요약/제목 메타 청크
    head = " ".join(first_text_lines)[:400]
    if head:
        docs.append(Document(
            page_content=f"[SUMMARY] {path.name} 개요: {head}",
            metadata={"source": str(path), "kind": "summary"},
        ))
    title = _extract_pdf_title(path)
    if title:
        docs.insert(0, Document(
            page_content=f"[TITLE] {title}",
            metadata={"source": str(path), "kind": "title"},
        ))
    return docs

# -----------------------------
# 디스패처
# -----------------------------
def load_docs_any(
    path: Union[str, Path],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    parser: str = "auto",
) -> List[Document]:
    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".pdf":
        docs = _load_pdf_auto(p)  # auto: 표-친화 시도 후 부족하면 기본과 병합
        return _chunk(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if ext == ".csv":
        docs = load_csv(p)
        return _chunk(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if ext == ".hwp":
        raise ValueError("HWP는 지원하지 않습니다.")
    if ext == ".pptx":
        docs = load_pptx(p)
    elif ext == ".xlsx":
        docs = load_xlsx(p)
    elif ext in (".txt", ".log", ".md"):
        docs = load_txt(p)
    else:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {ext}")

    return _chunk(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)