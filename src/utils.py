# src/utils.py
import os
import re
import httpx

# --- env (OpenAI 호환 내부 엔드포인트) ---
BASE_URL = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
API_KEY  = os.environ.get("OPENAI_API_KEY", "any")
MODEL    = os.environ.get("OPENAI_MODEL", "")

# --- public: drop_think ---
_think_pat = re.compile(r"<\s*think\b[^>]*>.*?</\s*think\s*>\s*", re.S | re.I)
def drop_think(text: str) -> str:
    """<think>…</think> 블록을 결과에서 제거"""
    return _think_pat.sub("", text or "")

# --- public: proxy_get ---
async def proxy_get(path: str = "/models", params: dict | None = None):
    """
    내부 OpenAI 호환 서버로 GET 프록시.
    예: await proxy_get("/models")
    """
    if not BASE_URL:
        raise RuntimeError("OPENAI_BASE_URL env 가 비어있습니다.")
    url = f"{BASE_URL}/{path.lstrip('/')}"
    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(url, params=params or {}, headers=headers)
        r.raise_for_status()
        return r.json()

# --- public: call_chat_completions ---
async def call_chat_completions(messages: list[dict], **kwargs):
    """
    내부 OpenAI 호환 서버의 /v1/chat/completions 호출.
    예:
      await call_chat_completions(
        messages=[{"role":"user","content":"안녕"}],
        temperature=0
      )
    """
    if not BASE_URL:
        raise RuntimeError("OPENAI_BASE_URL env 가 비어있습니다.")
    if not MODEL:
        raise RuntimeError("OPENAI_MODEL env 가 비어있습니다.")

    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": MODEL, "messages": messages}
    payload.update(kwargs)

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        return r.json()

__all__ = ["proxy_get", "call_chat_completions", "drop_think"]