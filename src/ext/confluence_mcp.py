# src/ext/confluence_mcp.py
from __future__ import annotations
import os, asyncio, logging
from typing import Any, Dict, List, Optional

from mcp import ClientSession
from mcp.client.sse import sse_client

log = logging.getLogger(__name__)

MCP_URL = os.getenv("MCP_CONFLUENCE_URL", "http://mcp-atlassian:9000/sse")

# 서버 구현마다 이름이 다를 수 있으니 후보군 준비
TOOL_CANDIDATES = (
    "confluence.search",
    "search_pages",
    "search",
    "confluence_search",
)

async def mcp_search(query: str, limit: int = 5, timeout: int = 20) -> List[Dict[str, Any]]:
    """
    MCP(Confluence) 서버의 '검색' 툴 호출 → [{title, url, body}] 리스트 반환.
    list_tools()는 일부 서버에서 파라미터 검증으로 실패할 수 있어 직접 후보 툴을 호출해본다.
    """
    results: List[Dict[str, Any]] = []
    picked: Optional[str] = None

    async with sse_client(MCP_URL) as (read, write):
        async with ClientSession(read, write) as session:
            # 1) 후보 툴명을 순차 호출해서 먼저 성공하는 걸 픽
            for name in TOOL_CANDIDATES:
                try:
                    _ = await asyncio.wait_for(
                        session.call_tool(name, {"q": "ping", "limit": 1}),
                        timeout=10
                    )
                    picked = name
                    break
                except Exception as e:
                    continue

            if not picked:
                raise RuntimeError(f"No working search tool found on MCP at {MCP_URL}. tried={TOOL_CANDIDATES}")

            # 2) 실제 질의
            resp = await asyncio.wait_for(
                session.call_tool(picked, {"q": query, "limit": limit}),
                timeout=timeout
            )

            for c in resp.content:
                ctype = getattr(c, "type", None)
                if ctype == "json":
                    payload = getattr(c, "json", None)
                    if isinstance(payload, list):
                        for it in payload:
                            results.append(_normalize_item(it))
                    elif isinstance(payload, dict):
                        results.append(_normalize_item(payload))
                elif ctype == "text":
                    txt = getattr(c, "text", "") or ""
                    if txt.strip():
                        results.append({"title": "", "url": "", "body": txt})
                else:
                    results.append({"title": "", "url": "", "body": str(c)})

    # 중복/빈 값 제거
    uniq: List[Dict[str, Any]] = []
    seen = set()
    for r in results:
        key = (
            (r.get("title","") or "").strip(),
            (r.get("url","")   or "").strip(),
            (r.get("body","")[:80] if r.get("body") else "")
        )
        if key in seen:
            continue
        seen.add(key)
        if (r.get("body") or "").strip():
            uniq.append(r)
    return uniq

def _normalize_item(d: Dict[str, Any]) -> Dict[str, Any]:
    title = (d.get("title") or d.get("name") or "").strip()
    url   = (d.get("url")   or d.get("link") or d.get("webui") or "").strip()
    body  = (d.get("body")  or d.get("content") or d.get("excerpt") or d.get("text") or "")
    if isinstance(body, dict):
        body = body.get("storage") or body.get("view") or body.get("plain") or body.get("value") or ""
        if isinstance(body, dict):
            body = body.get("value") or ""
    return {"title": title, "url": url, "body": str(body or "")}
