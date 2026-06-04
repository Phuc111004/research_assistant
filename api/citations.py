"""Build structured citation metadata from retrieved papers."""
from typing import Any, Dict, List


def _snippet(text: str, max_len: int = 220) -> str:
    if not text:
        return ""
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3].rstrip() + "..."


def build_citations(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One citation entry per retrieved paper (1-based index matches [1], [2] in answers)."""
    citations: List[Dict[str, Any]] = []
    for i, paper in enumerate(papers, start=1):
        abstract = paper.get("abstract") or ""
        citations.append(
            {
                "index": i,
                "paper_id": str(paper.get("paper_id", "")),
                "title": paper.get("title") or "Untitled Paper",
                "snippet": _snippet(abstract),
                "score": paper.get("score"),
                "doi": paper.get("doi"),
                "url": paper.get("url") or paper.get("pdf_url"),
            }
        )
    return citations


def default_citation_system_prompt() -> str:
    return (
        "You are a helpful research assistant that answers using ONLY the provided papers. "
        "Write a clear, structured answer in the same language as the user's question. "
        "After each factual claim or bullet point that comes from a specific paper, "
        "append an inline citation marker using the paper number, e.g. [1] or [2]. "
        "Use [1] for Paper 1, [2] for Paper 2, etc. You may cite multiple papers like [1][3]. "
        "Do not invent papers or facts not supported by the context."
    )
