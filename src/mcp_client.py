# src/mcp_client.py
from __future__ import annotations

import os
import requests
from typing import Any, Dict, List

MCP_URL = os.getenv("MCP_URL", "http://mcp-atlassian:9000/mcp").rstrip("/")
PROTO = os.getenv("MCP_PROTOCOL_VERSION", "2025-06-18")

# FastMCP(Streamable HTTP) 서버가 요구하는 헤더
BASE_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "MCP-Protocol-Version": PROTO,
}


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
        args: Dict[str, Any] = {"query": query, "limit": limit}
        if space:
            args["space"] = space

        res = self._post({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "confluence_search", "arguments": args},
        })

        # 결과 포맷은 서버 구현에 따라 다소 다를 수 있어 안전하게 파싱
        # 우선순위: result.content(list of dict/text) -> result.result -> result
        result = res.get("result", {}) or {}
        items = result.get("content") or result.get("result") or result.get("items") or []
        uris: List[str] = []
        for it in items if isinstance(items, list) else []:
            if isinstance(it, dict) and "uri" in it:
                uris.append(it["uri"])
            elif isinstance(it, str):
                uris.append(it)
        # 컨플 URI만
        return [u for u in uris if isinstance(u, str) and u.startswith("confluence://")]

    def read(self, uri: str) -> Dict[str, Any]:
        res = self._post({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "resources/read",
            "params": {"uri": uri},
        })
        result = res.get("result", {}) or {}
        text = result.get("text")
        if not text:
            # FastMCP 스타일: contents=[{type:'text', text:'...'}]
            contents = result.get("contents") or []
            if contents and isinstance(contents[0], dict):
                text = contents[0].get("text")
        meta = result.get("meta") or {}
        return {"uri": uri, "text": text or "", "meta": meta}
