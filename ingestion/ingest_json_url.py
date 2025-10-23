"""JSON (including OpenAPI) ingestor.

Fetches JSON from a URL, converts it into logically segmented documents,
chunks content, embeds with OpenAI embeddings, and stores Chunk rows in Postgres.

Capabilities:
- Detects OpenAPI (presence of 'openapi' or 'swagger' roots) vs generic JSON
- For OpenAPI:
  - Creates one logical document per (path + method) operation
  - Creates component documents (schemas/parameters/responses/requestBodies)
  - Attaches rich metadata via title/section/url fields
- For generic JSON:
  - Flattens to dot-path key/value lines
  - Groups by top-level object keys into documents

Chunking:
- Uses app.utils.chunk_text with settings.CHUNK_SIZE and settings.CHUNK_OVERLAP
- doc_id is deterministic per (source_url + section_key) to enable upserts/dedup

Usage:
  python -m app.ingestion.ingest_json_url --url https://example.com/openapi.json

Configuration:
- Database: app.config.settings.DATABASE_URL
- Embeddings: app.config.settings.OPENAI_EMBEDDING_MODEL
- Chunk params: app.config.settings.CHUNK_SIZE, CHUNK_OVERLAP
"""
from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict, Iterable, List, Tuple

logger = logging.getLogger(__name__)

import requests
from sqlalchemy.orm import Session

from app.config import settings
from app.db import init_db, session_scope
from app.models import Chunk
from app.embedding import embed_texts
from app.utils import (
    chunk_text,
    normalize_url,
    stable_doc_id,
)

HEADERS = {
    "User-Agent": "RAG-Ingestor/1.0 (+https://example.com; contact=dev@example.com)",
    "Accept": "application/json",
}


def fetch_json(url: str, timeout: int = 30) -> Dict[str, Any]:
    """Fetch JSON from a URL with basic headers and timeout."""
    logger.info("Fetching JSON: %s", url)
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    ctype = resp.headers.get("Content-Type", "")
    size = len(resp.content or b"")
    logger.info("HTTP %d from %s (content-type=%s, bytes=%d)", resp.status_code, url, ctype, size)
    resp.raise_for_status()
    # Try JSON regardless of content-type as long as body is JSON-decodable
    return resp.json()


def is_openapi(doc: Dict[str, Any]) -> bool:
    """Heuristic to detect OpenAPI/Swagger documents."""
    return isinstance(doc, dict) and ("openapi" in doc or "swagger" in doc) and ("paths" in doc)


def _schema_brief(schema: Any, max_len: int = 500) -> str:
    """Produce a compact human-readable summary for a schema object."""
    try:
        if isinstance(schema, dict):
            if "$ref" in schema:
                return f"$ref: {schema['$ref']}"
            t = schema.get("type")
            fmt = schema.get("format")
            if t:
                return f"type: {t}" + (f" ({fmt})" if fmt else "")
            # Fallback to a trimmed JSON
            s = json.dumps(schema, ensure_ascii=False)
            return (s[:max_len] + "…") if len(s) > max_len else s
        # Non-dict types: stringify safely
        s = json.dumps(schema, ensure_ascii=False)
        return (s[:max_len] + "…") if len(s) > max_len else s
    except Exception:
        s = str(schema)
        return (s[:max_len] + "…") if len(s) > max_len else s


def _render_parameters(params: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for p in params or []:
        name = p.get("name", "")
        loc = p.get("in", "")
        req = "required" if p.get("required") else "optional"
        desc = (p.get("description") or "").strip()
        schema = _schema_brief(p.get("schema"))
        lines.append(f"- {name} in {loc} ({req}) — {schema}" + (f"\n  {desc}" if desc else ""))
    return "\n".join(lines) if lines else "(none)"


def _render_request_body(rb: Dict[str, Any]) -> str:
    if not rb:
        return "(none)"
    mt = rb.get("content", {}) or {}
    parts: List[str] = []
    for media, spec in mt.items():
        schema = _schema_brief((spec or {}).get("schema"))
        examples = (spec or {}).get("examples") or {}
        example_count = len(examples)
        parts.append(f"- {media} — {schema}" + (f" (examples: {example_count})" if example_count else ""))
    return "\n".join(parts) if parts else "(none)"


def _render_responses(responses: Dict[str, Any]) -> str:
    if not responses:
        return "(none)"
    lines: List[str] = []
    for code, r in responses.items():
        desc = (r or {}).get("description") or ""
        mt = (r or {}).get("content") or {}
        if mt:
            for media, spec in mt.items():
                schema = _schema_brief((spec or {}).get("schema"))
                lines.append(f"- {code} {media} — {schema}" + (f"\n  {desc}" if desc else ""))
        else:
            lines.append(f"- {code}" + (f" — {desc}" if desc else ""))
    return "\n".join(lines)


def _operation_to_text(path: str, method: str, op: Dict[str, Any]) -> Tuple[str, str]:
    """Return (title, text) for an OpenAPI operation."""
    method_u = method.upper()
    title = f"{method_u} {path}"
    summary = (op.get("summary") or "").strip()
    description = (op.get("description") or "").strip()
    tags = op.get("tags") or []

    params = op.get("parameters") or []
    # Some specs also define path-level parameters; handle outside if needed.

    request_body = op.get("requestBody") or {}
    responses = op.get("responses") or {}

    text_parts: List[str] = [
        f"# {title}",
    ]
    if summary:
        text_parts.append(f"Summary: {summary}")
    if tags:
        text_parts.append(f"Tags: {', '.join(tags)}")
    if description:
        text_parts.append("\nDescription:\n" + description)

    text_parts.append("\nParameters:\n" + _render_parameters(params))
    text_parts.append("\nRequest Body:\n" + _render_request_body(request_body))
    text_parts.append("\nResponses:\n" + _render_responses(responses))

    return title, "\n".join(text_parts).strip()


def extract_openapi_sections(doc: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """Extract OpenAPI into section triplets: (section_key, title, text)."""
    out: List[Tuple[str, str, str]] = []

    # Operations
    paths: Dict[str, Any] = (doc.get("paths") or {})
    for path, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        for method, op in methods.items():
            # Only HTTP methods
            if method.lower() not in {"get", "post", "put", "delete", "patch", "head", "options", "trace"}:
                continue
            if not isinstance(op, dict):
                continue
            title, text = _operation_to_text(path, method, op)
            section_key = f"op::{method.upper()} {path}"
            out.append((section_key, title, text))

    # Components
    components: Dict[str, Any] = (doc.get("components") or {})
    for comp_type in ["schemas", "parameters", "responses", "requestBodies"]:
        comp_map: Dict[str, Any] = components.get(comp_type) or {}
        if not isinstance(comp_map, dict):
            continue
        for name, value in comp_map.items():
            title = f"component {comp_type}.{name}"
            # Dump a reasonably formatted JSON; chunker will split it anyway
            try:
                dumped = json.dumps(value, ensure_ascii=False, indent=2)
            except Exception:
                dumped = str(value)
            text = f"# {title}\n\n{dumped}"
            section_key = f"component::{comp_type}.{name}"
            out.append((section_key, title, text))

    return out


def flatten_json(value: Any, prefix: str = "") -> List[Tuple[str, str]]:
    """Flatten arbitrary JSON to (dot_path, scalar_str) pairs.

    For arrays, we index with [i]. For objects, we use dot notation.
    Non-scalar values are JSON-dumped to strings.
    """
    out: List[Tuple[str, str]] = []

    def _walk(v: Any, key_path: str):
        if isinstance(v, dict):
            for k, nv in v.items():
                kp = f"{key_path}.{k}" if key_path else str(k)
                _walk(nv, kp)
        elif isinstance(v, list):
            for i, nv in enumerate(v):
                kp = f"{key_path}[{i}]"
                _walk(nv, kp)
        else:
            # scalar
            try:
                if isinstance(v, (str, int, float, bool)) or v is None:
                    sval = json.dumps(v, ensure_ascii=False)
                else:
                    sval = json.dumps(v, ensure_ascii=False)
            except Exception:
                sval = str(v)
            out.append((key_path, sval))

    _walk(value, prefix)
    return out


def extract_generic_sections(doc: Any) -> List[Tuple[str, str, str]]:
    """Create sections grouped by top-level keys for a generic JSON document.

    Returns:
        List of (section_key, title, text)
    """
    sections: List[Tuple[str, str, str]] = []
    if isinstance(doc, dict):
        for k, v in doc.items():
            pairs = flatten_json(v, k)
            # Render as "path: value" lines
            lines = [f"{p}: {val}" for p, val in pairs]
            title = f"json section: {k}"
            text = "# " + title + "\n\n" + "\n".join(lines)
            sections.append((f"json::{k}", title, text))
    else:
        # Fallback: flatten everything as root
        pairs = flatten_json(doc, "root")
        lines = [f"{p}: {val}" for p, val in pairs]
        title = "json section: root"
        text = "# " + title + "\n\n" + "\n".join(lines)
        sections.append(("json::root", title, text))
    return sections


def _ingest_sections(db: Session, source_url: str, sections: List[Tuple[str, str, str]]) -> int:
    """Chunk, embed, and insert rows for provided sections.

    Args:
        db: SQLAlchemy session
        source_url: Original JSON URL
        sections: List of (section_key, title, text)

    Returns:
        Total number of chunks created.
    """
    created_total = 0
    base_url = normalize_url(source_url)

    for section_key, title, text in sections:
        # Chunk per section to keep doc_id consistent for this logical document
        chunks = chunk_text(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        if not chunks:
            continue
        logger.debug("Section %s => %d chunks", section_key, len(chunks))

        embeddings = embed_texts(chunks)
        # Deterministic doc_id per (source + section)
        doc_id = stable_doc_id(f"{base_url}::{section_key}")

        for i, (content, emb) in enumerate(zip(chunks, embeddings)):
            row = Chunk(
                doc_id=doc_id,
                url=f"{base_url}#{section_key}",
                title=(title or "")[:500] or None,
                section=(title or "")[:500] or None,
                position=i,
                content=content,
                embedding=emb,
            )
            db.add(row)
            created_total += 1

    return created_total


def ingest_json_url(url: str) -> int:
    """Top-level function: fetch, detect, extract, and persist."""
    data = fetch_json(url)
    try:
        if is_openapi(data):
            version_field = data.get("openapi") or data.get("swagger")
            version = version_field.strip() if isinstance(version_field, str) else version_field
            paths = data.get("paths") or {}
            num_paths = len(paths) if isinstance(paths, dict) else 0
            http_methods = {"get", "post", "put", "delete", "patch", "head", "options", "trace"}
            num_ops = 0
            if isinstance(paths, dict):
                for methods in paths.values():
                    if isinstance(methods, dict):
                        for m in methods.keys():
                            if isinstance(m, str) and m.lower() in http_methods:
                                num_ops += 1
            comp = data.get("components") or {}
            num_comp_items = 0
            if isinstance(comp, dict):
                for v in comp.values():
                    if isinstance(v, dict):
                        num_comp_items += len(v)
            logger.info(
                "Detected OpenAPI spec (version=%s): paths=%d, operations=%d, components=%d",
                version, num_paths, num_ops, num_comp_items
            )
            sections = extract_openapi_sections(data)
        else:
            top_keys = len(data) if isinstance(data, dict) else 1
            logger.info("Detected generic JSON: top_level_keys=%d, type=%s", top_keys, type(data).__name__)
            sections = extract_generic_sections(data)
        logger.info("Extracted %d sections from %s", len(sections), url)
    except Exception:
        logger.exception("Failed while analyzing JSON structure for %s", url)
        raise

    with session_scope() as db:
        created = _ingest_sections(db, url, sections)
    logger.info("Inserted %d chunks from %d sections (url=%s)", created, len(sections), url)
    return created


def main():
    parser = argparse.ArgumentParser(description="Ingest JSON (including OpenAPI) from a URL.")
    parser.add_argument("--url", required=True, help="JSON URL to ingest (e.g., an OpenAPI spec)")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger.info("Starting JSON ingestion for %s", args.url)

    init_db()
    try:
        total = ingest_json_url(args.url)
        logger.info("Completed ingestion: chunks=%d, url=%s", total, args.url)
        print(f"[INGEST-JSON] {args.url} -> {total} chunks")
    except Exception:
        logger.exception("Ingestion failed for %s", args.url)
        raise


if __name__ == "__main__":
    main()
