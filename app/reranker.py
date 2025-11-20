"""Cross-encoder reranking utilities.

Provides:
- _load_model: Lazy-load a sentence-transformers CrossEncoder for reranking.
- score_pairs: Score (query, passage) pairs; higher scores indicate stronger relevance.

The model name and enablement are configured via app.config.settings.
"""
from typing import List, Optional, Tuple

from app.config import settings

_model = None  # lazy-loaded to avoid container cold start cost


def _load_model():
    """Load and cache the cross-encoder reranker model.

    Returns:
        Any: A sentence-transformers CrossEncoder instance.
    """
    global _model
    if _model is not None:
        return _model
    # Lightweight cross-encoder; good quality/cost balance
    from sentence_transformers.cross_encoder import CrossEncoder

    name = settings.RERANKER_MODEL_NAME
    _model = CrossEncoder(name, trust_remote_code=True)
    return _model


def score_pairs(query: str, passages: List[str]) -> List[float]:
    """Score (query, passage) pairs for relevance using a cross-encoder.

    Args:
        query: The user query to compare against passages.
        passages: List of passages to score.

    Returns:
        List[float]: Relevance scores aligned with the input passages; higher is better.
    """
    if not passages:
        return []
    model = _load_model()
    pairs: List[Tuple[str, str]] = [(query, p) for p in passages]
    scores: List[float] = model.predict(pairs, convert_to_numpy=True).tolist()
    return scores
