# src/ext/confluence_mcp.py
from __future__ import annotations
import os, asyncio, json, logging
from typing import Any, Dict, List, Optional

from mcp import ClientSession
from mcp.client.sse import sse_client

log = logging.getLogger(__name__)

MCP_URL = os.getenv("MCP_CONFLUENCE_URL", "http://mcp-atlassian:9000/sse")
PREFERRED_TOOLS = {"confluence_search", "confluence_search_user"}


def _tool_name(item):
    # dict
    if isinstance(item, dict):
        return item.get("name")
    # (name, something) tuple/list
    if isinstance(item, (list, tuple)) and item:
        return item[0]
    # 객체 (dataclass 등)
    return getattr(item, "name", None)

async def mcp_search(
    query: str,
    limit: int = 5,
    timeout: int = 20,
    spaces_filter: Optional[str] = None,  # ← 필요 시 외부에서 주입
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    async with sse_client(MCP_URL) as (read, write):
        async with ClientSession(read, write) as session:
            # 1) 초기화
            await session.initialize()

            # 2) 툴 선택 (선호 툴 우선, 없으면 첫 번째 유효 이름으로 폴백)
            tools = await session.list_tools()
            names = [_tool_name(t) for t in tools]
            tool_name = next((n for n in names if n in PREFERRED_TOOLS), None) \
                        or next((n for n in names if n), None)
            if not tool_name:
                raise RuntimeError(f"No usable Confluence tool; got={names}")

            # 3) 호출 (한 번만)
            args = {
                "query": query,
                "limit": limit,
                "spaces_filter": spaces_filter if spaces_filter is not None
                                  else os.getenv("CONFLUENCE_SPACES_FILTER"),
            }
            resp = await session.call_tool(tool_name, args, timeout=timeout)

            # 4) 에러면 예외
            if getattr(resp, "is_error", False):
                msg = "; ".join(
                    getattr(c, "text", "") for c in getattr(resp, "content", []) 
                    if getattr(c, "type", None) == "text"
                ) or "MCP tool error"
                raise RuntimeError(msg)

            # 5) content 파싱
            for c in getattr(resp, "content", []):
                ctype = getattr(c, "type", None)
                if ctype == "json":
                    _collect_payload(results, getattr(c, "json", None))
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
