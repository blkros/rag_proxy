# src/retrieval/rerank.py
import re
from collections import defaultdict

ARTICLE_RE = re.compile(r'제\s*(\d{1,3})\s*조')
NUMSEC_RE  = re.compile(r'\d+(?:\.\d+)+')  # 1.2, 2.3.1 등

def parse_query_intent(q: str):
    q_norm = (q or "").strip()
    want_article = None
    m = ARTICLE_RE.search(q_norm)
    if m:
        try:
            want_article = int(m.group(1))
        except:
            pass
    want_numsec = NUMSEC_RE.search(q_norm) is not None
    is_summary  = any(k in q_norm for k in ("요약", "핵심", "중요", "정리", "5가지", "3가지"))
    return {"article_no": want_article, "want_numsec": want_numsec, "is_summary": is_summary}

def simple_keywords(q: str):
    q = (q or "").lower()
    for tok in re.split(r'\s+', q):
        tok = tok.strip(".,:()[]\"'")
        if len(tok) >= 2:
            yield tok

def rerank(query: str, hits: list[dict]) -> list[dict]:
    intent = parse_query_intent(query)
    want_article = intent["article_no"]
    for h in hits:
        md = h.get("metadata") or {}
        bonus = 0.0
        if md.get("kind") == "section":
            bonus += 0.06
        if want_article and md.get("article_no") == want_article:
            bonus += 0.20
        st = (md.get("section_title") or "").lower()
        if st:
            for kw in simple_keywords(query):
                if kw and kw in st:
                    bonus += 0.01
        h["rerank"] = float(h.get("score", 0.0)) + bonus
    hits.sort(key=lambda x: x.get("rerank", x.get("score", 0.0)), reverse=True)
    return hits

def coverage_sample(hits: list[dict], per_section=2, max_k=8) -> list[dict]:
    by_sec = defaultdict(list)
    for h in hits:
        idx = (h.get("metadata") or {}).get("section_index", -1)
        by_sec[idx].append(h)
    ordered_secs = sorted(
        by_sec.keys(),
        key=lambda si: (by_sec[si][0].get("rerank", by_sec[si][0].get("score", 0.0)) if by_sec[si] else -1),
        reverse=True,
    )
    out = []
    for si in ordered_secs:
        for h in by_sec[si][:per_section]:
            out.append(h)
            if len(out) >= max_k:
                return out
    return out

def pick_for_injection(query: str, hits: list[dict], k_default=8) -> list[dict]:
    intent = parse_query_intent(query)
    hits = rerank(query, hits)
    if intent["article_no"]:
        target = [h for h in hits if (h.get("metadata") or {}).get("article_no") == intent["article_no"]]
        if target:
            si = (target[0].get("metadata") or {}).get("section_index")
            neighbor = [h for h in hits if (h.get("metadata") or {}).get("section_index") == si]
            bag = (target + neighbor + hits)
            # 중복 제거(텍스트 기준 간단 dedup)
            seen = set()
            uniq = []
            for x in bag:
                key = (x["metadata"].get("source"), x["metadata"].get("page"), x["metadata"].get("section_index"), x["text"][:80])
                if key in seen: continue
                seen.add(key); uniq.append(x)
            return uniq[:k_default]
        return hits[:k_default]
    if intent["is_summary"]:
        return coverage_sample(hits, per_section=2, max_k=k_default)
    return hits[:k_default]