"""Stripe docs crawler and ingestor.

Crawls allowed Stripe documentation pages starting from configured seed URLs,
extracts text into sections, chunks content, embeds with OpenAI embeddings, and
stores Chunk rows into Postgres with pgvector.

Main functions:
- extract_links: find and normalize in-domain links from a page
- fetch: HTTP GET with basic headers and timeout
- ingest_page: parse, chunk, embed, and insert rows for a single page
- crawl_and_ingest: BFS crawl up to MAX_DOCS, ingesting each visited page

Configuration:
- Seeds, limits, chunk params: app.config.settings (DOCS_SEED_URLS, MAX_DOCS, CHUNK_SIZE, CHUNK_OVERLAP)
- Database: app.config.settings.DATABASE_URL
- Embeddings: app.config.settings.OPENAI_EMBEDDING_MODEL
"""
import time
import re
import queue
from typing import List, Set, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session

from app.config import settings
from app.db import init_db, session_scope
from app.models import Chunk
from app.embedding import embed_texts
from app.utils import (
    html_to_text_with_sections,
    chunk_sections,
    normalize_url,
    stable_doc_id,
    is_allowed_url,
)


HEADERS = {
    "User-Agent": "RAG-Ingestor/1.0 (+https://example.com; contact=dev@example.com)"
}


def is_toc_page(html: str, url: str) -> bool:
    """Heuristic detection of Table of Contents pages.

    Returns True if the page appears to primarily list links (TOC),
    in which case we should skip saving/chunking but still use it to discover links.
    """
    try:
        soup = BeautifulSoup(html, "lxml")

        # 1) Title/heading match
        title_text = ""
        if soup.title and soup.title.string:
            title_text = soup.title.string.strip().lower()
        h1 = soup.find("h1")
        h1_text = h1.get_text(" ", strip=True).strip().lower() if h1 else ""
        if title_text in {"table of contents", "contents"} or h1_text in {"table of contents", "contents"}:
            return True

        # 2) Structural cues in id/class
        pattern = re.compile(r"(?:^|\b)(toc|table-of-contents|contents)(?:\b|$)", re.I)
        for el in soup.find_all(["nav", "div", "ul", "aside"]):
            el_id = (el.get("id") or "")
            el_class = el.get("class") or []
            class_str = " ".join(el_class) if isinstance(el_class, list) else str(el_class)
            if (el_id and pattern.search(el_id)) or (class_str and pattern.search(class_str)):
                return True

        # 3) Content density (lots of links, little text or high link ratio)
        # links = soup.find_all("a", href=True)
        # link_count = len(links)
        # page_text = soup.get_text(" ", strip=True) or ""
        # text_len = len(page_text)
        # link_text_len = sum(len((a.get_text(" ", strip=True) or "")) for a in links)
        # link_ratio = (link_text_len / text_len) if text_len else 0.0
        # if link_count >= 20 and text_len < 1200:
        #     return True
        # if text_len > 0 and link_ratio >= 0.6:
        #     return True

        # 4) Section shape (few sections, very short total)
        # try:
        #     sections = html_to_text_with_sections(html)
        #     total_words = sum(len(txt.split()) for _, txt in sections)
        #     if len(sections) <= 3 and total_words <= 150:
        #         return True
        # except Exception:
        #     pass

        return False
    except Exception:
        return False


def extract_links(base_url: str, html: str) -> List[str]:
    """Extract and normalize in-domain links from an HTML page.

    Resolves relative links against base_url, normalizes URLs, and filters using
    is_allowed_url.

    Args:
        base_url: The URL used to resolve relative hrefs.
        html: The page HTML to parse.

    Returns:
        List[str]: Deduplicated list of absolute, normalized URLs.
    """
    soup = BeautifulSoup(html, "lxml")
    links: List[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Resolve relative links
        abs_url = urljoin(base_url, href)
        abs_url = normalize_url(abs_url)
        # Keep same origin and allowed domain
        if is_allowed_url(abs_url):
            links.append(abs_url)
    # De-dup while preserving order
    seen: Set[str] = set()
    out: List[str] = []
    for u in links:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def fetch(url: str) -> Tuple[int, str]:
    """Fetch a URL with a simple GET request.

    Args:
        url: Absolute URL to fetch.

    Returns:
        Tuple[int, str]: (HTTP status code, response text if ok else empty string).
    """
    resp = requests.get(url, headers=HEADERS, timeout=20)
    return resp.status_code, resp.text if resp.ok else ""


def ingest_page(db: Session, url: str, html: str) -> int:
    """Parse, chunk, embed and insert rows for a single page.

    Extracts a title, converts HTML into (section, text) blocks, chunks text,
    batches embeddings, and inserts Chunk rows.

    Args:
        db: SQLAlchemy session.
        url: Page URL (used for doc_id and metadata).
        html: Raw HTML content.

    Returns:
        int: Number of chunk rows created.
    """
    soup = BeautifulSoup(html, "lxml")
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()[:500]

    sections = html_to_text_with_sections(html)
    chunks = chunk_sections(sections)
    if not chunks:
        return 0

    texts = [c for (_, c) in chunks]
    embeddings = embed_texts(texts)
    doc_id = stable_doc_id(url)

    created = 0
    for i, ((section, content), emb) in enumerate(zip(chunks, embeddings)):
        row = Chunk(
            doc_id=doc_id,
            url=url,
            title=title,
            section=section[:500] if section else None,
            position=i,
            content=content,
            embedding=emb,
        )
        db.add(row)
        created += 1

    return created


def crawl_and_ingest():
    """Breadth-first crawl starting from configured seed URLs and ingest pages.

    Respects MAX_DOCS, normalizes/filters URLs, and uses a single session scope for
    efficiency. Prints progress and a final summary.
    """
    init_db()

    seed_urls = [normalize_url(u.strip()) for u in settings.DOCS_SEED_URLS.split(",") if u.strip()]
    max_docs = settings.MAX_DOCS

    q: "queue.Queue[str]" = queue.Queue()
    for u in seed_urls:
        q.put(u)

    visited: Set[str] = set()
    processed = 0
    t0 = time.time()

    with session_scope() as db:
        while not q.empty() and processed < max_docs:
            url = q.get()
            url = normalize_url(url)
            if url in visited:
                continue
            visited.add(url)

            try:
                status, html = fetch(url)
            except Exception:
                continue
            if status != 200 or not html:
                continue

            # Ingest page (skip Table of Contents pages)
            try:
                if is_toc_page(html, url):
                    print(f"[SKIP-TOC] {url}")
                else:
                    n = ingest_page(db, url, html)
                    processed += 1
                    print(f"[INGEST] {processed}/{max_docs} {url} -> {n} chunks")
            except Exception as e:
                print(f"[ERROR] ingest {url}: {e}")

            # Enqueue neighbors
            for nxt in extract_links(url, html):
                if nxt not in visited:
                    q.put(nxt)

    dt = time.time() - t0
    print(f"[DONE] Ingested {processed} pages in {dt:.1f}s")


if __name__ == "__main__":
    crawl_and_ingest()
