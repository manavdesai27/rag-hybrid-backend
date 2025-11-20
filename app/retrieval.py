"""Retrieval pipeline utilities for hybrid and reranked candidate selection.

This module implements:
- Tokenization and keyword term extraction
- SQL keyword prefilter builder
- Min-max normalization utility
- retrieve_candidates: hybrid vector + BM25 with optional cross-encoder reranking

Vector search uses pgvector cosine distance (similarity = 1 - distance).
BM25 scoring is applied within a merged candidate pool.
"""
import re
from typing import Dict, List, Tuple, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session
from rank_bm25 import BM25Okapi

from app.config import settings
from app.embedding import embed_query
from app.reranker import score_pairs


def _tokenize(s: str) -> List[str]:
    """Lowercase alphanumeric tokenization used by BM25 and term extraction.

    Args:
        s: Input string.

    Returns:
        List[str]: Alphanumeric tokens in lowercase.
    """
    return re.findall(r"[a-z0-9]+", s.lower())


def _extract_terms(question: str, min_len: int = 3, max_terms: int = 6) -> List[str]:
    """Extract distinctive keyword terms for SQL prefiltering.

    Selects unique terms of at least min_len characters, preferring longer tokens.

    Args:
        question: The user question text.
        min_len: Minimum token length to consider.
        max_terms: Maximum number of terms to return.

    Returns:
        List[str]: Ordered list of distinctive terms (longer first).
    """
    toks = _tokenize(question)
    # keep distinctive terms (longer first)
    toks = [t for t in toks if len(t) >= min_len]
    # de-dup preserving order
    seen = set()
    out: List[str] = []
    for t in sorted(toks, key=lambda x: (-len(x), x)):
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= max_terms:
            break
    return out


def _min_max_norm(xs: List[float]) -> List[float]:
    """Min-max normalize a list of scores to [0, 1].

    Args:
        xs: Sequence of numeric scores.

    Returns:
        List[float]: Normalized scores, or zeros if constant/empty.
    """
    if not xs:
        return []
    mn, mx = min(xs), max(xs)
    if mx - mn <= 1e-12:
        return [0.0 for _ in xs]
    return [(x - mn) / (mx - mn) for x in xs]


def _build_keyword_query(terms: List[str]) -> Tuple[str, Dict[str, str]]:
    """Build a simple ILIKE-based SQL prefilter for provided terms.

    Args:
        terms: List of keyword terms.

    Returns:
        Tuple[str, Dict[str, str]]: (SQL string with placeholders, params dict).
    """
    conds: List[str] = []
    params: Dict[str, str] = {}
    for i, t in enumerate(terms):
        key = f"t{i}"
        conds.append(f"content ILIKE :{key}")
        params[key] = f"%{t}%"
    where = " OR ".join(conds) if conds else "FALSE"
    sql = f"""
        SELECT id, url, title, section, content
        FROM chunks
        WHERE {where}
        LIMIT :limit
    """
    params["limit"] = 50
    return sql, params


def retrieve_candidates(db: Session, question: str, top_k: int, alpha_override: Optional[float] = None) -> Tuple[List[Dict], float]:
    """Hybrid retrieval combining vector, keyword/BM25, and optional cross-encoder.

    Args:
        db: SQLAlchemy session.
        question: User question string.
        top_k: Number of passages to return.
        alpha_override: Optional override for vector weight alpha in [0, 1].

    Returns:
        Tuple[List[Dict], float]: A tuple of:
            - selected: top_k items with fields including id, url, title, section,
              content, snippet, score, bm25, vec_norm.
            - best_vec_sim: best raw vector similarity (1 - distance) from initial search.

    Notes:
        Steps:
        1) Vector search via pgvector cosine distance (similarity = 1 - distance).
        2) Keyword prefilter and BM25 scoring within merged candidate pool.
        3) Min-max normalization and alpha-weighted blend.
        4) Optional cross-encoder reranking on top-N if enabled.
    """
    qvec = embed_query(question)
    qvec_str = "[" + ",".join(f"{x:.6f}" for x in qvec) + "]"

    # 1) Vector search: distance lower is better; similarity = 1 - distance
    vec_sql = text(
        """
        SELECT id, url, title, section, content, 
            (embedding <=> CAST(:qvec AS vector)) AS distance
        FROM chunks
        ORDER BY embedding <=> CAST(:qvec AS vector)
        LIMIT :limit
        """
    )
    vec_rows = db.execute(vec_sql, {"qvec": qvec_str, "limit": 80}).mappings().all()
    vec_items = []
    for r in vec_rows:
        dist = float(r["distance"])
        sim = max(0.0, 1.0 - dist)
        vec_items.append(
            {
                "id": int(r["id"]),
                "url": r["url"],
                "title": r["title"],
                "section": r["section"],
                "content": r["content"],
                "vec_sim": sim,
            }
        )

    best_vec_sim = max((it["vec_sim"] for it in vec_items), default=0.0)

    # 2) Keyword fetch
    terms = _extract_terms(question)
    kw_items: List[Dict] = []
    if terms:
        kw_sql_str, kw_params = _build_keyword_query(terms)
        kw_rows = db.execute(text(kw_sql_str), kw_params).mappings().all()
        for r in kw_rows:
            kw_items.append(
                {
                    "id": int(r["id"]),
                    "url": r["url"],
                    "title": r["title"],
                    "section": r["section"],
                    "content": r["content"],
                    "vec_sim": 0.0,  # unknown
                }
            )

    # Merge candidates by id (prioritize vector fields)
    by_id: Dict[int, Dict] = {}
    for it in vec_items + kw_items:
        cur = by_id.get(it["id"])
        if cur is None or it.get("vec_sim", 0.0) > cur.get("vec_sim", 0.0):
            by_id[it["id"]] = it
    pool: List[Dict] = list(by_id.values())
    if not pool:
        return [], best_vec_sim

    # 3) BM25 within pool
    docs = [d["content"] for d in pool]
    tokenized = [_tokenize(x) for x in docs]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = list(bm25.get_scores(_tokenize(question)))  # arbitrary scale

    # Normalize scores and combine
    vec_scores = [d.get("vec_sim", 0.0) for d in pool]
    vec_norm = _min_max_norm(vec_scores)
    bm_norm = _min_max_norm(bm25_scores)

    alpha = float(alpha_override) if alpha_override is not None else 0.6  # vector weight
    combined = [alpha * v + (1 - alpha) * b for v, b in zip(vec_norm, bm_norm)]
    for d, v, b, c in zip(pool, vec_norm, bm_norm, combined):
        d["score"] = float(c)
        d["bm25"] = float(b)
        d["vec_norm"] = float(v)

    # Optional cross-encoder reranking on top-N
    if settings.RERANKER_ENABLED:
        topn = min(settings.RERANKER_TOPN, len(pool))
        head = pool[:topn]
        passages = [d["content"] for d in head]
        try:
            rerank_scores = score_pairs(question, passages)
            rr_norm = _min_max_norm([float(s) for s in rerank_scores])
            beta = 0.55  # blend weight toward reranker
            for d, rr in zip(head, rr_norm):
                d["score"] = beta * rr + (1 - beta) * d["score"]
            pool = sorted(pool, key=lambda x: x["score"], reverse=True)
        except Exception:
            # If reranker model load/predict fails, continue without it
            pass

    # --- DEBUGGING OUTPUT ---
    print("---- DEBUG: Candidate Retrieval Pool ----")
    for idx, cand in enumerate(pool[:25]):
        print(f"Rank {idx+1}: URL={cand['url']}, vec_sim={cand.get('vec_sim', 0):.3f}, bm25={cand.get('bm25',0):.3f}, score={cand.get('score',0):.3f}, title={cand.get('title')}")
    print(f"Total candidates in pool: {len(pool)}\n")

    # Sort and select top_k
    pool.sort(key=lambda x: x["score"], reverse=True)
    selected = pool[:top_k]

    # Construct concise snippets for citations
    for d in selected:
        content = d["content"].strip()
        if len(content) > 450:
            content = content[:450].rstrip() + "..."
        d["snippet"] = content

    # Determine hybrid top_score: use top result's score, or 0.0 if no candidates
    top_score = selected[0]["score"] if selected else 0.0

    return selected, top_score
