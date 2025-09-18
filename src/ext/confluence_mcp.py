# src/ext/confluence_mcp.py
from __future__ import annotations
import os, asyncio, json, logging, hashlib
from typing import Any, Dict, List, Optional

from mcp import ClientSession
from mcp.client.sse import sse_client  # << 변경 포인트

log = logging.getLogger(__name__)

MCP_URL = os.getenv("MCP_CONFLUENCE_URL", "http://mcp-confluence:9000/mcp")

async def mcp_search(query: str, limit: int = 5, timeout: int = 20) -> List[Dict[str, Any]]:
    """
    MCP(Confluence) 서버의 '검색' 툴 호출 → [{title, url, body}] 리스트 반환.
    """
    url = MCP_URL
    results: List[Dict[str, Any]] = []

    # SSE로 MCP 세션 연결 (신 SDK는 sse_client가 정답)
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            tools = await session.list_tools()
            # 서버 구현 이름 폭넓게 매칭
            cand = ("confluence.search", "search_pages", "search", "confluence_search")
            tool_name: Optional[str] = next((t.name for t in tools if t.name in cand), None)
            if not tool_name:
                raise RuntimeError(f"No search tool found on MCP at {url}. tools={[t.name for t in tools]}")

            # 호출에 타임아웃 적용
            resp = await asyncio.wait_for(
                session.call_tool(tool_name, {"q": query, "limit": limit}),
                timeout=timeout
            )

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
                    results.append({"title": "", "url": "", "body": str(c)})

    # 중복/빈 값 정리
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
