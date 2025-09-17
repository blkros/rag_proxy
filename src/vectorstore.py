# src/vectorstore.py
from __future__ import annotations
from typing import Optional, Any

# 다른 모듈들이 참조할 공유 핸들
vectorstore: Optional[Any] = None
retriever: Optional[Any] = None
