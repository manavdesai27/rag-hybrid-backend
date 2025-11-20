"""Answer generation utilities using OpenAI chat completions.

Provides:
- get_client: Cached OpenAI client
- _build_context: Formatting of retrieved passages into a compact context block
- generate_answer: Grounded answer generation constrained to provided context

Configuration is read from app.config.settings.
"""
from typing import Dict, List
from openai import OpenAI

from app.config import settings

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Return a cached OpenAI Chat Completions client using the configured API key.

    Returns:
        OpenAI: Client instance reused across calls.
    """
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


def _build_context(cands: List[Dict]) -> str:
    """Create a compact, enumerated context block from retrieved candidates.

    Args:
        cands: Retrieved passages; each item should contain section/title, url, and snippet/content.

    Returns:
        str: Human-readable block with [n] source | url and snippet text.
    """
    lines: List[str] = []
    for i, c in enumerate(cands, start=1):
        source = c.get("section") or c.get("title") or ""
        url = c.get("url", "")
        snippet = c.get("snippet", c.get("content", "")) or ""
        lines.append(f"[{i}] {source} | {url}\n{snippet}")
    return "\n\n".join(lines)


def generate_answer(question: str, candidates: List[Dict], max_tokens: int | None = None) -> str:
    """Generate an answer grounded strictly in provided candidate passages.

    The model is instructed to avoid fabrications and to not include inline citations.

    Args:
        question: User question to answer.
        candidates: Retrieved passages to ground the answer.
        max_tokens: Optional cap for output tokens; defaults to settings.MAX_OUTPUT_TOKENS.

    Returns:
        str: The generated answer text.
    """
    client = get_client()
    context = _build_context(candidates)

    system = (
        "You are a helpful assistant answering questions about technical API/product documentation. "
        "Use ONLY the provided context passages to answer. If the answer is not clearly supported, say you don't know. "
        "Be factual and complete. Do NOT include citation markers in your answer; citations will be attached by the system."
    )
    user = (
        f"Question:\n{question}\n\n"
        f"Context passages (use these only):\n{context}\n\n"
        "Provide a clear and complete answer grounded in the context. If unsure, say you don't know."
    )

    resp = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=max_tokens or settings.MAX_OUTPUT_TOKENS,
    )
    content = resp.choices[0].message.content or ""
    return content.strip()
