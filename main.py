"""FastAPI application entrypoint and routes.

Exposes health and /ask endpoints, configures CORS, and initializes the database
schema at startup. The /ask endpoint orchestrates routing, retrieval, generation,
observability, and caching.
"""
import time
from typing import List

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.cache import get_cached_answer, set_cached_answer
from app.config import settings
from app.db import get_db, init_db
from app.generation import generate_answer
from app.retrieval import retrieve_candidates
from app.schemas import AskRequest, AskResponse, Citation
from app.router import classify_query
from app.obs import Trace, span
from app.rate_limit import rag_rate_limited

app = FastAPI(title="RAG API", version="0.1.0")

# Allow UI (localhost:8501) and any dev origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # keep simple for demo; tighten for prod
    allow_methods=["*"],
    allow_credentials=True,
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    """Initialize database schema and indexes at application startup."""
    # Ensure DB schema and indexes exist
    init_db()


@app.get("/health")
def health():
    """Liveness probe endpoint.

    Returns:
        dict: {"status": "ok"} when the service is running.
    """
    return {"status": "ok"}


@rag_rate_limited(
    question_key=lambda args, kwargs: (
        (kwargs.get("req") or (args[0] if args else None)).question.strip().lower()
        + "|max="
        + str((kwargs.get("req") or (args[0] if args else None)).max_tokens)
    )
)
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request, db: Session = Depends(get_db)) -> AskResponse:
    """Answer a user question using retrieval-augmented generation.

    Workflow:
    - Check Redis cache for existing answer keyed by question+max_tokens
    - Classify query to decide hybrid weighting
    - Retrieve candidates (vector + keyword/BM25, optional reranker)
    - Refuse if similarity too low or nothing found
    - Generate grounded answer using OpenAI with provided context
    - Cache the result and return citations and latency

    Args:
        req: AskRequest payload.
        db: Database session (FastAPI dependency).

    Returns:
        AskResponse: Answer, citations, latency, and cache flag.
    """
    t0 = time.time()
    trace = Trace("ask", input={"question": req.question})

    # Check cache
    cached = get_cached_answer(req.question, req.max_tokens)
    if cached:
        # Don't trust cached latency; measure fresh request handling time
        latency_ms = int((time.time() - t0) * 1000)
        citations = [Citation(**c) for c in cached.get("citations", [])]
        try:
            trace.event("cache_hit", {"latency_ms": latency_ms})
            trace.end(output={"used_cache": True, "latency_ms": latency_ms})
        except Exception:
            pass
        return AskResponse(
            answer=cached.get("answer", ""),
            citations=citations,
            latency_ms=latency_ms,
            used_cache=True,
        )

    # Route and retrieve candidates
    decision = classify_query(req.question)
    trace.event("route", {"route": decision.route, "reason": decision.reason, "alpha": decision.alpha})
    with span("retrieve", {"route": decision.route, "alpha": decision.alpha}):
        cands, top_score = retrieve_candidates(
            db, req.question, settings.TOP_K, alpha_override=decision.alpha
        )
    trace.event("retrieval_result", {"num_candidates": len(cands), "top_score": top_score})

    # Refusal if hybrid score is too low or nothing found
    if not cands or top_score < settings.REFUSAL_MIN_SIMILARITY:
        msg = (
            "I don't know. The retrieved documentation does not confidently support an answer."
        )
        latency_ms = int((time.time() - t0) * 1000)
        resp = AskResponse(
            answer=msg,
            citations=[],
            latency_ms=latency_ms,
            used_cache=False,
        )
        # Cache refusal to avoid repeated work
        set_cached_answer(
            req.question,
            {"answer": resp.answer, "citations": [c.dict() for c in resp.citations]},
            req.max_tokens,
        )
        try:
            trace.event("refusal", {"top_score": top_score})
            trace.end(output={"used_cache": False, "latency_ms": latency_ms})
        except Exception:
            pass
        return resp

    # Generate grounded answer
    answer = generate_answer(req.question, cands, max_tokens=req.max_tokens)

    # Build citations from top candidates
    citations: List[Citation] = []
    for c in cands[: settings.TOP_K]:
        source = c.get("section") or c.get("title") or None
        citations.append(
            Citation(
                source=source,
                url=c.get("url", ""),
                snippet=c.get("snippet", c.get("content", "")) or "",
            )
        )

    resp = AskResponse(
        answer=answer,
        citations=citations,
        latency_ms=int((time.time() - t0) * 1000),
        used_cache=False,
    )

    # Observability
    try:
        trace.generation("answer", prompt=req.question, output=resp.answer, metadata={"candidates": len(cands)})
        trace.end(output={"used_cache": False, "latency_ms": resp.latency_ms})
    except Exception:
        pass

    # Cache the response (without latency/used_cache)
    set_cached_answer(
        req.question, {"answer": resp.answer, "citations": [c.dict() for c in citations]}, req.max_tokens
    )

    return resp
