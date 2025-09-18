# src/ext/confluence_mcp.py
from __future__ import annotations
import os, asyncio, logging
from typing import Any, Dict, List

from mcp import ClientSession
from mcp.client.sse import sse_client

log = logging.getLogger(__name__)

MCP_URL = os.getenv("MCP_CONFLUENCE_URL", "http://mcp-atlassian:9000/sse")
TOOL_CANDIDATES = ("confluence_search", "confluence.search", "search_pages", "search")

async def mcp_search(query: str, limit: int = 5, timeout: int = 20) -> List[Dict[str, Any]]:
    """
    Confluence MCP의 검색 툴 호출 -> [{title, url, body}]
    - SSE 연결 후 반드시 initialize를 먼저 수행
    - 서버마다 인자 이름이 다를 수 있어 query/q 둘 다 시도
    """
    results: List[Dict[str, Any]] = []

    async with sse_client(MCP_URL) as (read, write):
        async with ClientSession(read, write) as session:
            # 1) MCP handshake (이거 안 하면 'before initialization' 에러 남)
            await session.initialize()

            # 2) 동작하는 툴 고르기 (우선 confluence_search)
            picked = None
            for name in TOOL_CANDIDATES:
                try:
                    # 일부 서버는 'ping' 같은 질의를 싫어할 수 있으니 바로 실제 query로 검증
                    await asyncio.wait_for(
                        session.call_tool(name, {"query": query, "limit": 1}),
                        timeout=10
                    )
                    picked = name
                    break
                except Exception:
                    # 'query' 키워드가 아닐 수 있어 'q'로도 재시도
                    try:
                        await asyncio.wait_for(
                            session.call_tool(name, {"q": query, "limit": 1}),
                            timeout=10
                        )
                        picked = name
                        break
                    except Exception:
                        continue

            if not picked:
                raise RuntimeError(f"No working search tool found on MCP at {MCP_URL}. tried={TOOL_CANDIDATES}")

            # 3) 실제 검색 호출 (query -> 실패시 q로 재시도)
            try:
                resp = await asyncio.wait_for(
                    session.call_tool(picked, {"query": query, "limit": limit}),
                    timeout=timeout
                )
            except Exception:
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
                    txt = (getattr(c, "text", "") or "").strip()
                    if txt:
                        results.append({"title": "", "url": "", "body": txt})
                else:
                    results.append({"title": "", "url": "", "body": str(c)})

    # 중복/빈값 제거
    uniq: List[Dict[str, Any]] = []
    seen = set()
    for r in results:
        key = ((r.get("title") or "").strip(),
               (r.get("url") or "").strip(),
               (r.get("body") or "")[:80])
        if key in seen:
            continue
        seen.add(key)
        if (r.get("body") or "").strip():
            uniq.append(r)
    return uniq

def _normalize_item(d: Dict[str, Any]) -> Dict[str, Any]:
    title = (d.get("title") or d.get("name") or "").strip()
    url   = (d.get("url") or d.get("link") or d.get("webui") or "").strip()
    body  = (d.get("body") or d.get("content") or d.get("excerpt") or d.get("text") or "")
    if isinstance(body, dict):
        body = body.get("storage") or body.get("view") or body.get("plain") or body.get("value") or ""
        if isinstance(body, dict):
            body = body.get("value") or ""
    return {"title": title, "url": url, "body": str(body or "")}
