# src/config.py
from __future__ import annotations
import os, json
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

load_dotenv(override=True)

def _as_list(val: str) -> List[str]:
    raw = (val or "").strip()
    if not raw:
        return []
    if raw == "*":
        return ["*"]
    return [x.strip() for x in raw.split(",") if x.strip()]

def _as_bool(val: str | None, default=False) -> bool:
    s = (val or "").strip().lower()
    if not s:
        return default
    return s in {"1","true","yes","y","on"}

@dataclass(frozen=True)
class Settings:
    # -------------------------
    # RAG / 인덱스 / 임베딩
    # -------------------------
    INDEX_DIR: str = field(default_factory=lambda: os.getenv("INDEX_DIR", "faiss_index"))
    UPLOAD_DIR: str = field(default_factory=lambda: os.getenv("UPLOAD_DIR", "uploads"))
    DATA_CSV: str = field(default_factory=lambda: os.getenv("DATA_CSV", "data/민생.csv"))
    CHUNK_SIZE: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "800")))
    CHUNK_OVERLAP: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "120")))
    EMBEDDING_MODEL: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large-instruct"))
    EMBEDDING_DEVICE: str = field(default_factory=lambda: os.getenv("EMBEDDING_DEVICE", "cpu"))  # or cuda
    E5_USE_PREFIX: bool = field(default_factory=lambda: _as_bool(os.getenv("E5_USE_PREFIX"), True))

    # -------------------------
    # 검색 튜닝
    # -------------------------
    RETRIEVER_K: int = field(default_factory=lambda: int(os.getenv("RETRIEVER_K", "5")))
    RETRIEVER_FETCH_K: int = field(default_factory=lambda: int(os.getenv("RETRIEVER_FETCH_K", "40")))
    SEARCH_TYPE: str = field(default_factory=lambda: os.getenv("SEARCH_TYPE", "mmr"))

    # -------------------------
    # 하이브리드(경량 스파스) + 타이틀/스페이스 힌트
    # -------------------------
    ENABLE_SPARSE: bool = field(default_factory=lambda: _as_bool(os.getenv("ENABLE_SPARSE"), True))
    SPARSE_LIMIT: int = field(default_factory=lambda: int(os.getenv("SPARSE_LIMIT", "150")))
    TITLE_BONUS: float = field(default_factory=lambda: float(os.getenv("TITLE_BONUS", "0.30")))
    SPACE_FILTER_MODE: str = field(default_factory=lambda: os.getenv("SPACE_FILTER_MODE", "soft"))  # "soft"|"hard"|"off"
    SPACE_HINT_BONUS: float = field(default_factory=lambda: float(os.getenv("SPACE_HINT_BONUS", "0.25")))
    PAGE_FILTER_MODE: str = field(default_factory=lambda: os.getenv("PAGE_FILTER_MODE", "soft"))    # "soft"|"hard"|"off"
    PAGE_HINT_BONUS: float = field(default_factory=lambda: float(os.getenv("PAGE_HINT_BONUS", "0.35")))
    SINGLE_SOURCE_COALESCE: bool = field(default_factory=lambda: _as_bool(os.getenv("SINGLE_SOURCE_COALESCE"), True))
    COALESCE_THRESHOLD: float = field(default_factory=lambda: float(os.getenv("COALESCE_THRESHOLD", "0.55")))
    ACRONYM_TITLE_BONUS: float = field(default_factory=lambda: float(os.getenv("ACRONYM_TITLE_BONUS", "0.45")))
    ACRONYM_BODY_BONUS: float = field(default_factory=lambda: float(os.getenv("ACRONYM_BODY_BONUS", "0.25")))

    # -------------------------
    # Sticky(연속 질문 안정화)
    # -------------------------
    STICKY_SECS: int = field(default_factory=lambda: int(os.getenv("STICKY_SECS", "180")))
    STICKY_STRICT: bool = field(default_factory=lambda: _as_bool(os.getenv("STICKY_STRICT"), True))
    STICKY_FROM_COALESCE: bool = field(default_factory=lambda: _as_bool(os.getenv("STICKY_FROM_COALESCE"), False))
    STICKY_AFTER_MCP: bool = field(default_factory=lambda: _as_bool(os.getenv("STICKY_AFTER_MCP"), True))
    STICKY_MODE: str = field(default_factory=lambda: os.getenv("STICKY_MODE", "bonus"))  # "bonus"|"filter"
    STICKY_BONUS: float = field(default_factory=lambda: float(os.getenv("STICKY_BONUS", "0.18")))
    PER_SOURCE_CAP: int = field(default_factory=lambda: int(os.getenv("PER_SOURCE_CAP", "0")))      # 0=off

    # -------------------------
    # 로컬 업로드 가중치/차단
    # -------------------------
    LOCAL_FIRST: bool = field(default_factory=lambda: _as_bool(os.getenv("LOCAL_FIRST"), True))
    LOCAL_BONUS: float = field(default_factory=lambda: float(os.getenv("LOCAL_BONUS", "0.25")))
    ALLOW_PDF_CONTEXT: bool = field(default_factory=lambda: _as_bool(os.getenv("ALLOW_PDF_CONTEXT"), False))
    UPLOAD_BLOCK_PREFIX: str = field(default_factory=lambda: os.getenv("UPLOAD_BLOCK_PREFIX", "uploads/"))

    # -------------------------
    # 도메인/스페이스 동적 힌트
    # -------------------------
    DOMAIN_PURITY_THRESHOLD: float = field(default_factory=lambda: float(os.getenv("DOMAIN_PURITY_THRESHOLD", "0.6")))
    DOMAIN_MIN_STRONG_TOKENS: int = field(default_factory=lambda: int(os.getenv("DOMAIN_MIN_STRONG_TOKENS", "1")))
    SPACE_SCORE_MIN: int = field(default_factory=lambda: int(os.getenv("SPACE_SCORE_MIN", "2")))

    # -------------------------
    # MCP / 폴백 / 타임존
    # -------------------------
    DISABLE_INTERNAL_MCP: bool = field(default_factory=lambda: _as_bool(os.getenv("DISABLE_INTERNAL_MCP"), False))
    MCP_WRITEBACK: bool = field(default_factory=lambda: _as_bool(os.getenv("MCP_WRITEBACK"), False))
    MCP_WRITEBACK_TITLES_ONLY: bool = field(default_factory=lambda: _as_bool(os.getenv("MCP_WRITEBACK_TITLES_ONLY"), False))
    MAX_FALLBACK_SECS: int = field(default_factory=lambda: int(os.getenv("MAX_FALLBACK_SECS", "7")))
    MCP_TIMEOUT: int = field(default_factory=lambda: int(os.getenv("MCP_TIMEOUT", "5")))
    MCP_MAX_TASKS: int = field(default_factory=lambda: int(os.getenv("MCP_MAX_TASKS", "4")))
    TZ_NAME: str = field(default_factory=lambda: os.getenv("TZ_NAME", "Asia/Seoul"))
    SEARCH_LANGS: List[str] = field(default_factory=lambda: _as_list(os.getenv("SEARCH_LANGS", "ko,en")))
    CONFLUENCE_BASE_URL: str = field(default_factory=lambda: os.getenv("CONFLUENCE_BASE_URL", "").rstrip("/"))

    # -------------------------
    # 재랭커
    # -------------------------
    ENABLE_RERANKER: bool = field(default_factory=lambda: _as_bool(os.getenv("ENABLE_RERANKER"), False))
    RERANKER_MODEL: str = field(default_factory=lambda: os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base"))
    RERANKER_TOP_N: int = field(default_factory=lambda: int(os.getenv("RERANKER_TOP_N", "5")))

    # -------------------------
    # LLM (vLLM/OpenAI 호환)
    # -------------------------
    OPENAI_API_KEY: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", "local-anything"))
    OPENAI_BASE_URL: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "http://172.16.10.168:9993/v1"))
    OPENAI_MODEL: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "qwen3-30b-a3b-fp8"))

    # -------------------------
    # 네트워크/보안/호스트 화이트리스트
    # -------------------------
    CORS_ORIGINS: List[str] = field(default_factory=lambda: _as_list(os.getenv("CORS_ORIGINS", "*")))
    ALLOWED_SOURCE_HOSTS: List[str] = field(default_factory=lambda: _as_list(os.getenv("ALLOWED_SOURCE_HOSTS", "")))

    # -------------------------
    # PDF 파싱 튜닝
    # -------------------------
    PDF_TABLE_THRESHOLD: int = field(default_factory=lambda: int(os.getenv("PDF_TABLE_THRESHOLD", "5")))

settings = Settings()