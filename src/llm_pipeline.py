# src/llm_pipeline.py
from __future__ import annotations

import os
import requests
from typing import List

BASE = os.getenv("OPENAI_BASE_URL", "").rstrip("/")  # 예: http://172.16.10.168:9993/v1
KEY = os.getenv("OPENAI_API_KEY", "local-anything")
MODEL = os.getenv("OPENAI_MODEL", "qwen3-30b-a3b-fp8")
TEMP = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "1024"))


def chat_with_context(question: str, contexts: List[str], system_prompt: str) -> str:
    ctx_blob = "\n\n".join(f"[DOC{i+1}]\n{c}" for i, c in enumerate(contexts))
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{question}\n\n# Context\n{ctx_blob}"},
    ]
    r = requests.post(
        f"{BASE}/chat/completions",
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        json={"model": MODEL, "messages": messages, "temperature": TEMP, "max_tokens": MAX_TOKENS},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]
