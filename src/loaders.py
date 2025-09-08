# src/loaders.py
from __future__ import annotations
from pathlib import Path
from typing import List, Union
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import chardet
from pypdf import PdfReader
from pptx import Presentation
import openpyxl
import os

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

def load_txt(path: Path) -> List[Document]:
    raw = path.read_bytes()
    enc = chardet.detect(raw).get("encoding") or "utf-8"
    text = raw.decode(enc, errors="ignore")
    return [Document(page_content=text, metadata={"source": str(path), "type": "txt"})]

def load_csv(path: Path) -> List[Document]:
    import csv
    docs: List[Document] = []
    rows: List[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        try:
            rdr = csv.DictReader(f)
            for row in rdr:
                key = (row.get("항목") or row.get("key") or "").strip()
                val = (row.get("내용") or row.get("value") or "").strip()
                text = f"{key}: {val}" if key else val
                if text:
                    rows.append(text)
        except Exception:
            f.seek(0)
            rdr = csv.reader(f)
            for r in rdr:
                line = ",".join("" if v is None else str(v) for v in r).strip()
                if line:
                    rows.append(line)
    if rows:
        docs.append(Document(
            page_content="\n".join(rows),
            metadata={"source": str(path), "type": "csv"}
        ))
    return docs

def _extract_pdf_title(path: Path) -> str | None:
    # 1) PDF 메타데이터 title
    try:
        meta = PdfReader(str(path)).metadata or {}
        t = getattr(meta, "title", None)
        if t and str(t).strip():
            return str(t).strip()
    except Exception:
        pass
    # 2) 첫 페이지 첫 줄(텍스트) 휴리스틱
    try:
        with pdfplumber.open(str(path)) as pdf:
            if pdf.pages:
                text = pdf.pages[0].extract_text() or ""
                for ln in (l.strip() for l in text.splitlines()):
                    if len(ln) >= 3:
                        return ln[:120]
    except Exception:
        pass
    # 3) 폴백: 파일명(확장자 제거)
    return path.stem

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

def _load_pdf_auto(path: Path) -> List[Document]:
    """표→문장 + 텍스트 스니펫을 함께 만들고,
       표 문장이 너무 적으면 기존 load_pdf()로 폴백하거나 병합."""
    try:
        import pdfplumber
    except Exception:
        # pdfplumber가 없으면 그냥 기존 파서 사용
        return load_pdf(path)

    docs: List[Document] = []
    first_text_lines: list[str] = []
    table_sentence_cnt = 0
    text_snippet_cnt = 0

    with pdfplumber.open(str(path)) as pdf:
        for pi, page in enumerate(pdf.pages, start=1):
            # 1) 표 추출 → 문장화
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

            # 2) 텍스트 스니펫(보강)
            text = page.extract_text() or ""
            lines = [_norm(l) for l in text.splitlines() if _norm(l)]
            if pi == 1:
                first_text_lines = lines[:10]
            if lines:
                snippet = " ".join(lines[:20])[:1200]
                if snippet:
                    text_snippet_cnt += 1
                    docs.append(Document(
                        page_content=f"[SOURCE:{path.name}] {snippet}",
                        metadata={"source": str(path), "page": pi, "kind": "page_text"},
                    ))

    # 3) 표가 거의 없으면 기존 파서 결과와 병합(또는 완전 폴백)
    threshold = int(os.getenv("PDF_TABLE_THRESHOLD", "5"))
    if table_sentence_cnt < threshold:
        legacy = load_pdf(path)
        docs = _dedup(legacy + docs)  # 병합(중복 제거)
        # 완전 폴백 원하면 위 한 줄을 'docs = legacy'로 교체

    # 4) 요약 청크 1개(메타질의 방어)
    head = " ".join(first_text_lines)[:400]
    if head:
        docs.append(Document(
            page_content=f"[SUMMARY] {path.name} 개요: {head}",
            metadata={"source": str(path), "kind": "summary"},
        ))

    # 5) 제목 청크 추가 (신규)
    title = _extract_pdf_title(path)
    if title:
        docs.insert(0, Document(
            page_content=f"[TITLE] {title}",
            metadata={"source": str(path), "kind": "title"},
        ))
    return docs

def load_docs_any(
    path: Union[str, Path],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    parser: str = "auto",        # ← 기본값은 auto
) -> List[Document]:
    p = Path(path)
    ext = p.suffix.lower()

    # ===== PDF 자동 감지 경로 =====
    if ext == ".pdf":
        if parser == "pdf_table":
            docs = _load_pdf_auto(p)   # 강제 표 친화
            return _chunk(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            # auto: 먼저 표 친화 시도 → 필요 시 자동 폴백/병합
            docs = _load_pdf_auto(p)
            return _chunk(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # ===== 기존 로직 그대로 =====
    if ext == ".hwp":
        raise ValueError("HWP는 지원하지 않습니다.")
    if ext == ".pptx":
        docs = load_pptx(p)
    elif ext == ".xlsx":
        docs = load_xlsx(p)
    elif ext in (".txt", ".log", ".md"):
        docs = load_txt(p)
    elif ext == ".csv":
        docs = load_csv(p)  # CSV 로더가 있다면
    else:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {ext}")

    return _chunk(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)