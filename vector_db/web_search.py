from __future__ import annotations
from typing import List

def web_search_fallback(query: str, max_results: int = 3) -> str:
    """RAG kanıtı bulunamazsa web'de ara."""
    try:
        from ddgs import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(
                f"{query} ilaç farmakoloji",
                region="tr-tr",
                max_results=max_results
            ):
                results.append(f"[{r['title']}] {r['body']}")
        return "\n\n".join(results) if results else ""
    except Exception as e:
        return ""