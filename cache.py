"""Caching utilities for answers using Redis.

Provides:
- get_redis: Cached Redis client from REDIS_URL with decode_responses.
- _key_for_question: Stable cache key derived from question + max_tokens.
- get_cached_answer: Fetch cached JSON answer for a question.
- set_cached_answer: Store JSON answer with TTL from settings.CACHE_TTL_SECONDS.
"""
import json
import hashlib
from typing import Any, Optional

import redis

from app.config import settings

_redis_client: Optional[redis.Redis] = None


def get_redis() -> redis.Redis:
    """Return a cached Redis client configured from settings.REDIS_URL.

    Returns:
        redis.Redis: Client with decode_responses=True.
    """
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis_client


def _key_for_question(question: str, max_tokens: Optional[int] = None) -> str:
    """Compute a stable cache key for a question and token cap.

    Args:
        question: User question string.
        max_tokens: Optional max tokens; defaults to settings.MAX_OUTPUT_TOKENS in key.

    Returns:
        str: Namespaced cache key.
    """
    norm_q = question.strip().lower()
    mt = max_tokens or settings.MAX_OUTPUT_TOKENS
    h = hashlib.sha256(f"{norm_q}|max={mt}".encode("utf-8")).hexdigest()
    return f"rag:qa:v2:{h}"


def get_cached_answer(question: str, max_tokens: Optional[int] = None) -> Optional[dict]:
    """Get a cached answer payload for the given question if present.

    Args:
        question: User question.
        max_tokens: Optional token cap dimensioning the cache key.

    Returns:
        Optional[dict]: Parsed JSON payload if found and valid; otherwise None.
    """
    r = get_redis()
    raw = r.get(_key_for_question(question, max_tokens))
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def set_cached_answer(question: str, value: dict, max_tokens: Optional[int] = None) -> None:
    """Store an answer payload under the computed cache key with TTL.

    Args:
        question: User question.
        value: JSON-serializable dict to store.
        max_tokens: Optional token cap that affects the cache key.
    """
    r = get_redis()
    r.setex(_key_for_question(question, max_tokens), settings.CACHE_TTL_SECONDS, json.dumps(value))
