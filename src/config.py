# src/config.py
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

load_dotenv(override=True)  # .env를 OS 환경변수로 로드 (없으면 무시)

def _as_list(val: str) -> List[str]:
    # "a,b , c" -> ["a","b","c"]; "*" -> ["*"]
    raw = (val or "").strip()
    if not raw:
        return []
    if raw == "*":
        return ["*"]
    return [x.strip() for x in raw.split(",") if x.strip()]

@dataclass(frozen=True)
class Settings:
    # RAG / 인덱스
    INDEX_DIR: str = field(default_factory=lambda: os.getenv("INDEX_DIR", "faiss_index"))
    UPLOAD_DIR: str = field(default_factory=lambda: os.getenv("UPLOAD_DIR", "uploads"))
    DATA_CSV: str = field(default_factory=lambda: os.getenv("DATA_CSV", "data/민생.csv"))
    CHUNK_SIZE: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "800")))
    CHUNK_OVERLAP: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "120")))
    EMBEDDING_MODEL: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small"))
    EMBEDDING_DEVICE: str = field(default_factory=lambda: os.getenv("EMBEDDING_DEVICE", "cpu"))  # or cuda

    # LLM (vLLM/OpenAI 호환)
    OPENAI_API_KEY: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", "local-anything"))
    OPENAI_BASE_URL: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "http://localhost:9999/v1"))
    OPENAI_MODEL: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "qwen3-30b-a3b-fp8"))

    # Web
    CORS_ORIGINS: List[str] = field(default_factory=lambda: _as_list(os.getenv("CORS_ORIGINS", "*")))

    # 표 위주 pdf, 만약 pdf에 표 내용이 거의 없으면 기본 파서랑 병합
    PDF_TABLE_THRESHOLD: int = field(
        default_factory=lambda: int(os.getenv("PDF_TABLE_THRESHOLD", "5"))
    )

settings = Settings()