# rag-proxy/src/mcp_client.py
from __future__ import annotations

import os, re
import json
import requests
from typing import Any, Dict, List

MCP_URL = os.getenv("MCP_URL", "http://mcp-atlassian:9000/mcp").rstrip("/")
PROTO = os.getenv("MCP_PROTOCOL_VERSION", "2025-06-18")

# [# ADDED] Confluence Space 강제 제한(없으면 None)
CONF_SPACE = os.getenv("CONFLUENCE_SPACE") or os.getenv("CONF_DEFAULT_SPACE") or None

# FastMCP(Streamable HTTP) 서버가 요구하는 헤더
BASE_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "MCP-Protocol-Version": PROTO,
}

_META_PATTERNS = (
    r"^\s*###\s*task",            # ### Task:
    r"json\s*format",             # JSON format:
    r"<\s*chat_history\s*>",      # <chat_history>
    r"follow[-\s]*ups?",          # follow-up(s)
    r"title\s+with\s+an\s+emoji", # title with an emoji
    r"tags\s+categorizing",       # tags categorizing
    r"^query:\s*history",         # Query: History:
    r"^\s*history:",              # History:
    # (원하면 한국어 키워드도 추가 가능: r"가이드라인|출력|대화\s*기록|태그")
)

def _is_meta_query(q: str) -> bool:  # [ADDED]
    s = (q or "").lower()
    return any(re.search(p, s) for p in _META_PATTERNS)

def _new_session_url(base: str) -> str:
    """
    POST /mcp -> 307 + Location: /mcp/<session-id>/events 로 세션 생성
    반환된 Location이 절대경로가 아니면 절대경로로 보정
    """
    r = requests.post(base, headers=BASE_HEADERS, json={}, allow_redirects=False, timeout=10)
    loc = r.headers.get("Location")
    if not loc:
        raise RuntimeError(f"No session Location from MCP (status={r.status_code})")
    if loc.startswith("/"):
        root = base.split("/mcp")[0]
        loc = root + loc
    return loc


class MCP:
    """
    최소 MCP 클라이언트. 한 인스턴스가 하나의 세션 URL을 보유.
    """

    def __init__(self, url: str | None = None, protocol: str | None = None):
        self._base = (url or MCP_URL).rstrip("/")
        if protocol:
            BASE_HEADERS["MCP-Protocol-Version"] = protocol
        self.session = _new_session_url(self._base)

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(self.session, headers=BASE_HEADERS, json=payload, timeout=60)
        r.raise_for_status()
        # FastMCP는 SSE 프레이밍도 쓰지만, 기본 응답은 JSON 한 건임
        return r.json()

    # ---- 기본 핸드셰이크/인스펙션 ----
    def initialize(self) -> Dict[str, Any]:
        return self._post({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": BASE_HEADERS["MCP-Protocol-Version"],
                "capabilities": {"roots": {"listChanged": True}},
                "clientInfo": {"name": "rag-proxy", "version": "0.1.0"},
            },
        })

    def tools_list(self) -> Dict[str, Any]:
        return self._post({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})

    # ---- Confluence 검색/읽기 ----
    def search(self, query: str, limit: int = 5, space: str | None = None) -> List[str]:
        if _is_meta_query(query):
            return []
        args: Dict[str, Any] = {"query": query, "limit": limit}
        # [# CHANGED] space가 지정되지 않았을 때도 환경변수 강제 적용
        if space or CONF_SPACE:  # [# ADDED]
            args["space"] = space or CONF_SPACE  # [# ADDED]
        tool = os.getenv("CONFLUENCE_TOOL_SEARCH", "search_pages")
        res = self._post({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": tool, "arguments": args},
        })

        # 결과 포맷은 서버 구현에 따라 다소 다를 수 있어 안전하게 파싱
        result = res.get("result", {}) or {}
        items = (
            result.get("content")
            or result.get("contents")
            or result.get("result")
            or result.get("items")
            or []
        )

        if isinstance(items, list) and items and isinstance(items[0], dict) and ("type" in items[0]):
            flat = []
            for c in items:
                if c.get("type") == "json" and isinstance(c.get("json"), list):
                    flat.extend(c["json"])
                elif c.get("type") == "text" and isinstance(c.get("text"), str):
                    txt = c["text"].strip()
                    if txt.startswith("{") or txt.startswith("["):
                        try:
                            maybe = json.loads(txt)
                            if isinstance(maybe, list):
                                flat.extend(maybe)
                        except Exception:
                            pass
            items = flat or items

        uris: List[str] = []
        for it in items if isinstance(items, list) else []:
            if isinstance(it, dict) and "uri" in it:
                uris.append(it["uri"])
            elif isinstance(it, str):
                uris.append(it)

        return [u for u in uris if isinstance(u, str) and u.startswith("confluence://")]

    def read(self, uri: str) -> dict:
        res = self._post({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "resources/read",
            "params": {"uri": uri},
        })
        result = res.get("result", {}) or {}
        text = result.get("text") or result.get("contents", [{}])[0].get("text", "")
        meta = result.get("meta") or {}
        return {"uri": uri, "text": text or "", "meta": meta}