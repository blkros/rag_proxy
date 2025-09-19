# src/ext/confluence_mcp.py
from __future__ import annotations
import os, asyncio, json, logging
from typing import Any, Dict, List, Optional

from mcp import ClientSession
from mcp.client.sse import sse_client

log = logging.getLogger(__name__)

MCP_URL = os.getenv("MCP_CONFLUENCE_URL", "http://mcp-atlassian:9000/sse")
PREFERRED_TOOLS = ("confluence_search",)  # 명시

async def mcp_search(query: str, limit: int = 5, timeout: int = 20) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    async with sse_client(MCP_URL) as (read, write):
        async with ClientSession(read, write) as session:
            # 1) 초기화 반드시 수행
            await session.initialize()

            # 2) 툴 선택
            tools = await session.list_tools()
            tool_name: Optional[str] = next((t.name for t in tools if t.name in PREFERRED_TOOLS), None)
            if not tool_name:
                raise RuntimeError(f"No suitable tool found on MCP at {MCP_URL}. tools={[t.name for t in tools]}")

            # 3) 반드시 'query' 파라미터만 사용
            resp = await asyncio.wait_for(
                session.call_tool(tool_name, {"query": query, "limit": limit}),
                timeout=timeout
            )

            # 4) 에러 응답이면 바로 예외 (텍스트 메시지 추출)
            if getattr(resp, "is_error", False):
                msg = "; ".join(
                    getattr(c, "text", "") for c in resp.content
                    if getattr(c, "type", None) == "text"
                ) or "MCP tool error"
                raise RuntimeError(msg)

            # 5) content 파싱: json 우선, text가 JSON처럼 생기면 로드
            for c in resp.content:
                ctype = getattr(c, "type", None)
                if ctype == "json":
                    payload = getattr(c, "json", None)
                    _collect_payload(results, payload)
                elif ctype == "text":
                    txt = (getattr(c, "text", "") or "").strip()
                    payload = None
                    if txt.startswith("{") or txt.startswith("["):
                        try:
                            payload = json.loads(txt)
                        except Exception:
                            payload = None
                    if payload is not None:
                        _collect_payload(results, payload)
                    elif txt:
                        results.append({"title": "", "url": "", "body": txt})
                else:
                    # 기타 타입은 문자열화
                    results.append({"title": "", "url": "", "body": str(c)})

    return _dedup(results)

def _collect_payload(out: List[Dict[str, Any]], payload: Any) -> None:
    if isinstance(payload, list):
        for it in payload:
            out.append(_normalize_item(it))
    elif isinstance(payload, dict):
        out.append(_normalize_item(payload))

def _normalize_item(d: Dict[str, Any]) -> Dict[str, Any]:
    title = (d.get("title") or d.get("name") or "").strip()
    url   = (d.get("url")   or d.get("link") or d.get("webui") or "").strip()
    body  =  d.get("body")  or d.get("content") or d.get("excerpt") or d.get("text") or ""
    if isinstance(body, dict):
        body = body.get("storage") or body.get("view") or body.get("plain") or body.get("value") or ""
        if isinstance(body, dict):
            body = body.get("value") or ""
    return {"title": title, "url": url, "body": str(body or "")}

def _dedup(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for r in items:
        key = ((r.get("title","") or "").strip(),
               (r.get("url","") or "").strip(),
               (r.get("body","") or "")[:120])
        if key in seen: 
            continue
        seen.add(key)
        if (r.get("body") or "").strip():
            out.append(r)
    return out
