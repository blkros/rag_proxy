# rag-proxy/src/ext/confluence_mcp.py
from __future__ import annotations
import os, asyncio, json, logging
from typing import Any, Dict, List

from mcp import ClientSession
from mcp.client.sse import sse_client

log = logging.getLogger(__name__)

# ### CHANGED: 우선 MCP_URL → 그다음 MCP_CONFLUENCE_URL → 기본값(/sse?version=...)
MCP_URL = (
    os.getenv("MCP_URL")
    or os.getenv("MCP_CONFLUENCE_URL")
    or "http://mcp-confluence:9000/sse?version=2025-06-18"
)

# ### NEW: 프로토콜 버전 환경변수 지원 (없으면 2025-06-18)
MCP_PROTOCOL_VERSION = os.getenv("MCP_PROTOCOL_VERSION", "2025-06-18")

TOOL_OVERRIDE = (os.getenv("CONFLUENCE_MCP_TOOL", "search_pages") or "search_pages").strip()


def _tool_name(item):
    # dict
    if isinstance(item, dict):
        return item.get("name")
    # (name, something) tuple/list
    if isinstance(item, (list, tuple)) and item:
        return item[0]
    # 객체 (dataclass 등)
    return getattr(item, "name", None)

# ### NEW: 신/구 MCP SDK 호환 레이어
def _wrap_streams(read, write):
    """
    mcp.client.sse.sse_client 가 신버전에서는 anyio Stream(read.receive / write.send)을,
    구버전에서는 callables(read(), write(msg))를 반환한다.
    여기서 둘 다 callables 로 맞춰준다.
    """
    # 신버전: MemoryObjectReceiveStream / MemoryObjectSendStream 스타일
    if hasattr(read, "receive") and hasattr(write, "send"):
        async def _read():
            return await read.receive()
        async def _write(msg):
            return await write.send(msg)
        return _read, _write
    # 구버전: 이미 callables
    return read, write


async def mcp_search(query: str, limit: int = 5, timeout: int = 20) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    async with sse_client(MCP_URL) as (read, write):
        # ### NEW: 신/구 SDK 모두 동작하게 스트림을 callables 로 래핑
        read, write = _wrap_streams(read, write)

        async with ClientSession(read, write) as session:
            # ### CHANGED: 가능하면 프로토콜 버전 명시, 구버전이면 무시
            try:
                await session.initialize(protocol_version=MCP_PROTOCOL_VERSION)  # 신버전
            except TypeError:
                await session.initialize()  # 구버전 호환

            tool_name = TOOL_OVERRIDE

            resp = await asyncio.wait_for(
                session.call_tool(tool_name, {"query": query, "limit": limit}),
                timeout=timeout
            )

            if getattr(resp, "is_error", False):
                msg = "; ".join(
                    getattr(c, "text", "") for c in resp.content
                    if getattr(c, "type", None) == "text"
                ) or "MCP tool error"
                raise RuntimeError(msg)

            for c in resp.content:
                ctype = getattr(c, "type", None)
                if ctype == "json":
                    _collect_payload(results, getattr(c, "json", None))
                elif ctype == "text":
                    txt = (getattr(c, "text", "") or "").strip()
                    if txt.startswith("{") or txt.startswith("["):
                        try:
                            _collect_payload(results, json.loads(txt))
                            continue
                        except Exception:
                            pass
                    if txt:
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
    title = (d.get("title") or d.get("name") or "").strip() if isinstance(d, dict) else ""
    url   = (d.get("url")   or d.get("link") or d.get("webui") or "").strip() if isinstance(d, dict) else ""
    body  =  (d.get("body") or d.get("content") or d.get("excerpt") or d.get("text") or "") if isinstance(d, dict) else ""
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
