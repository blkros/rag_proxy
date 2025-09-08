# src/loaders.py
from __future__ import annotations
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import chardet
from pypdf import PdfReader
from pptx import Presentation
import openpyxl

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

def load_docs_any(path: Path, chunk_size=800, chunk_overlap=120) -> List[Document]:
    ext = path.suffix.lower()
    if ext == ".hwp":
        raise ValueError("HWP는 지원하지 않습니다.")
    if ext == ".pdf":
        docs = load_pdf(path)
    elif ext == ".pptx":
        docs = load_pptx(path)
    elif ext == ".xlsx":
        docs = load_xlsx(path)
    elif ext in (".txt", ".log", ".md"):
        docs = load_txt(path)
    else:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {ext}")
    return _chunk(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)