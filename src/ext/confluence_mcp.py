# src/ext/confluence_mcp.py
from __future__ import annotations
import os, asyncio, json, logging
from typing import Any, Dict, List, Optional

# MCP SDK (https://github.com/modelcontextprotocol/python-sdk)
from mcp.client.session import ClientSession
from mcp.client.sse import sse_connect

log = logging.getLogger(__name__)

MCP_URL = os.getenv("MCP_CONFLUENCE_URL", "http://mcp-confluence:9000/mcp")

async def mcp_search(query: str, limit: int = 5, timeout: int = 20) -> List[Dict[str, Any]]:
    """MCP(Confluence) 서버의 '검색' 툴 호출 → [{title, url, body}] 리스트 반환."""
    url = MCP_URL
    results: List[Dict[str, Any]] = []

    # SSE로 MCP 세션 연결
    async with sse_connect(url) as conn:
        async with ClientSession(conn) as session:
            # 사용 가능한 툴 목록
            tools = await session.list_tools()
            tool_name: Optional[str] = None
            # 툴 이름은 서버 구현에 따라 다를 수 있으니 넓게 매칭
            for t in tools:
                if t.name in ("confluence.search", "search_pages", "search"):
                    tool_name = t.name
                    break
            if not tool_name:
                raise RuntimeError(f"No search tool found on MCP at {url}. tools={[t.name for t in tools]}")

            # 툴 호출
            resp = await session.call_tool(tool_name, arguments={"q": query, "limit": limit})

            # 응답 content 파싱(text/json 모두 처리)
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
                    # 혹시 모를 다른 타입은 문자열화
                    results.append({"title": "", "url": "", "body": str(c)})

    # 중복/빈 값 정리
    uniq = []
    seen = set()
    for r in results:
        key = (r.get("title","").strip(), r.get("url","").strip(), (r.get("body","")[:80] if r.get("body") else ""))
        if key in seen: 
            continue
        seen.add(key)
        if r.get("body","").strip():
            uniq.append(r)
    return uniq

def _normalize_item(d: Dict[str, Any]) -> Dict[str, Any]:
    title = (d.get("title") or d.get("name") or "").strip()
    url   = (d.get("url") or d.get("link") or d.get("webui") or "").strip()
    body  = (d.get("body") or d.get("content") or d.get("excerpt") or d.get("text") or "")
    if isinstance(body, dict):
        # 혹시 body가 dict이면 가장 그럴듯한 필드 집계
        body = body.get("storage") or body.get("view") or body.get("plain") or body.get("value") or ""
        if isinstance(body, dict):
            body = body.get("value") or ""
    body = str(body)
    return {"title": title, "url": url, "body": body}
