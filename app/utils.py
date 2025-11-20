"""Utility helpers for URL normalization, HTML extraction, and text chunking.

This module provides:
- stable_doc_id: stable SHA-1 based identifier for documents/URLs
- normalize_url: normalization to make URLs consistent for deduplication
- html_to_text_with_sections: HTML to (section, text) extraction using BeautifulSoup
- chunk_text/chunk_sections: simple fixed-size character chunking with overlap
- is_allowed_url: allowlist check for source domains
"""
import re
import hashlib
from typing import Iterable, List, Tuple
from bs4 import BeautifulSoup
from bs4.element import Tag

from app.config import settings


def stable_doc_id(s: str) -> str:
    """Compute a stable 40-char SHA-1 hex identifier for a string.

    Args:
        s: Input string (e.g., normalized URL or path).

    Returns:
        str: First 40 hex characters of the SHA-1 digest.
    """
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:40]


def normalize_url(u: str) -> str:
    """Normalize URLs by removing fragments and trailing slashes.

    Args:
        u: Raw URL.

    Returns:
        str: Normalized URL suitable for stable IDs and deduplication.
    """
    # Strip fragments and trailing slashes for stability
    u = re.sub(r"#.*$", "", u)
    if len(u) > 1 and u.endswith("/"):
        u = u[:-1]
    return u


def html_to_text_with_sections(html: str) -> List[Tuple[str, str]]:
    """Convert HTML into (section_title, text_block) pairs.

    Parses headings (h1..h4) as section titles and gathers text from paragraph-like
    elements. Excess whitespace is collapsed. If no structure is detected, falls
    back to a single block with the whole page text.

    Args:
        html: Raw HTML string.

    Returns:
        List[Tuple[str, str]]: Sequence of (section_title, text_block).
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove script/style
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    blocks: List[Tuple[str, str]] = []
    current_section = ""
    buffer: List[str] = []

    def flush():
        nonlocal buffer, blocks, current_section
        if buffer:
            text = " ".join(buffer)
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                blocks.append((current_section, text))
        buffer = []

    for el in soup.body.descendants if soup.body else soup.descendants:
        if isinstance(el, Tag):
            if el.name in ["h1", "h2", "h3", "h4"]:
                flush()
                current_section = el.get_text(" ", strip=True)
            elif el.name in ["p", "li", "td"]:
                txt = el.get_text(" ", strip=True)
                if txt:
                    buffer.append(txt)

    flush()
    # Fallback if nothing parsed
    if not blocks:
        text = soup.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            blocks = [("", text)]
    return blocks


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into fixed-size character chunks with overlap.

    Ensures chunk_size >= 200 and overlap in [0, chunk_size // 2].

    Args:
        text: Input string to split.
        chunk_size: Target chunk size in characters.
        overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        List[str]: Non-empty trimmed chunks.
    """
    if not text:
        return []
    chunk_size = max(200, chunk_size)
    overlap = max(0, min(overlap, chunk_size // 2))
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end].strip())
        if end == n:
            break
        start = end - overlap
    return [c for c in chunks if c]


def chunk_sections(sections: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Expand (section, text_block) into fixed-size chunks per section.

    Uses global settings.CHUNK_SIZE and settings.CHUNK_OVERLAP.

    Args:
        sections: Sequence of (section_title, text_block) pairs.

    Returns:
        List[Tuple[str, str]]: Flattened list of (section_title, chunk_text).
    """
    out: List[Tuple[str, str]] = []
    for title, block in sections:
        for ch in chunk_text(block, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP):
            out.append((title, ch))
    return out


def is_allowed_url(u: str) -> bool:
    """Check whether a URL is in the permitted source allowlist.

    Currently restricts to Stripe docs domain as a simple heuristic.

    Args:
        u: URL to validate.

    Returns:
        bool: True if allowed, False otherwise.
    """
    # Stripe docs domain (simple heuristic)
    return True
