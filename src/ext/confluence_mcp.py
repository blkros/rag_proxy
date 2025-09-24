# rag-proxy/src/ext/confluence_mcp.py
from __future__ import annotations
import os, asyncio, json, logging, re
from typing import Any, Dict, List

from mcp import ClientSession
from mcp.client.sse import sse_client

log = logging.getLogger(__name__)

MCP_PROTOCOL_VERSION = os.getenv("MCP_PROTOCOL_VERSION", "2025-06-18")

def _with_version(url: str) -> str:
    if not url:
        return f"http://mcp-confluence:9000/sse?version={MCP_PROTOCOL_VERSION}"
    if "version=" in url:
        return url
    sep = "&" if ("?" in url) else "?"
    return f"{url}{sep}version={MCP_PROTOCOL_VERSION}"

RAW_MCP_URL = (os.getenv("MCP_URL") or os.getenv("MCP_CONFLUENCE_URL") or "http://mcp-confluence:9000/sse")
MCP_URL = _with_version(RAW_MCP_URL)
TOOL_OVERRIDE = (os.getenv("CONFLUENCE_MCP_TOOL", "") or "").strip()
_LOGIN_PAT = re.compile(r"(Confluence에\s*로그인|로그인\s*-\s*Confluence|name=[\"']os_username[\"'])", re.I)

def _looks_like_login(s: str) -> bool:
    return bool(_LOGIN_PAT.search(s or ""))

def _tool_name(item):
    # dict
    if isinstance(item, dict):
        return item.get("name")
    # (name, something) tuple/list
    if isinstance(item, (list, tuple)) and item:
        return item[0]
    # 객체 (dataclass 등)
    return getattr(item, "name", None)

async def mcp_search(query: str, limit: int = 5, timeout: int = 20) -> List[Dict[str, Any]]:
    results_all: List[Dict[str, Any]] = []

    # 툴 실행 순서를 결정: override가 있으면 그거만, 없으면 search → search_pages
    if TOOL_OVERRIDE in ("search", "search_pages"):
        tool_order = [TOOL_OVERRIDE]
    else:
        tool_order = ["search", "search_pages"]

    async with sse_client(MCP_URL) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            for tool_name in tool_order:
                try:
                    resp = await asyncio.wait_for(
                        session.call_tool(tool_name, {"query": query, "limit": limit}),
                        timeout=timeout
                    )
                except Exception as e:
                    log.warning("MCP tool '%s' call failed: %s", tool_name, e)
                    continue

                if getattr(resp, "is_error", False):
                    msg = "; ".join(
                        getattr(c, "text", "") for c in resp.content
                        if getattr(c, "type", None) == "text"
                    ) or f"MCP tool error: {tool_name}"
                    log.warning(msg)
                    continue

                tmp: List[Dict[str, Any]] = []
                for c in resp.content:
                    ctype = getattr(c, "type", None)
                    if ctype == "json":
                        _collect_payload(tmp, getattr(c, "json", None))
                    elif ctype == "text":
                        txt = (getattr(c, "text", "") or "").strip()
                        if txt.startswith("{") or txt.startswith("["):
                            try:
                                _collect_payload(tmp, json.loads(txt)); continue
                            except Exception:
                                pass
                        if txt:
                            tmp.append({"title": "", "url": "", "body": txt})
                    else:
                        tmp.append({"title": "", "url": "", "body": str(c)})

                # 로그인 화면/빈 텍스트 제외
                tmp = [r for r in tmp if (r.get("body") or "").strip() and not _looks_like_login(r.get("body"))]
                results_all.extend(tmp)

                # search에서 이미 결과가 있으면 더 안 내려가도 됨
                if results_all:
                    break

    return _dedup(results_all)

def _collect_payload(out: List[Dict[str, Any]], payload: Any) -> None:
    if isinstance(payload, list):
        for it in payload:
            out.append(_normalize_item(it))
    elif isinstance(payload, dict):
        out.append(_normalize_item(payload))

def _normalize_item(d: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(d, dict):
        return {"id":"", "space":"", "version":0, "title":"", "url":"", "body": str(d)}

    title = (d.get("title") or d.get("name") or "").strip()
    url   = (d.get("url") or d.get("link") or d.get("webui") or "").strip()
    body  =  (d.get("body") or d.get("content") or d.get("excerpt") or d.get("text") or "")

    # body가 dict면 value 계층을 파고 들어감
    if isinstance(body, dict):
        body = body.get("storage") or body.get("view") or body.get("plain") or body.get("value") or ""
        if isinstance(body, dict):
            body = body.get("value") or ""
    body = str(body or "")
    # 하이라이트 태그 제거
    body = re.sub(r"@@@(?:hl|endhl)@@@", "", body)

    # 메타 보존: id/space/version
    pid = str(d.get("id") or d.get("page") or "").strip()
    space = (d.get("space") or d.get("spaceKey") or "").strip()
    try:
        version = int(d.get("version") or d.get("ver") or 0)
    except Exception:
        version = 0

    # URL에서 pageId 보강
    if (not pid) and url:
        m = re.search(r"[?&]pageId=(\d+)", url)
        if m:
            pid = m.group(1)

    return {
        "id": pid,
        "space": space,
        "version": version,
        "title": title,
        "url": url,
        "body": body,
    }

def _dedup(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for r in items:
        body = (r.get("body") or "").strip()
        # 본문이 없어도 허용. 로그인 화면만 제외
        if _looks_like_login(body):
            continue
        key = (
            (r.get("id") or "").strip(),
            (r.get("url") or "").strip(),
            (r.get("title") or "").strip(),
            body[:160],
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out