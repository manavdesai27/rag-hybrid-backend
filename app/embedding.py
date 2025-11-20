"""Embedding utilities wrapping OpenAI's embeddings API.

Provides:
- get_client: Cached OpenAI client using the configured API key.
- embed_texts: Batch embedding for a list of strings.
- embed_query: Convenience helper to embed a single query string.

Models and dimensions are configured via app.config.settings.
"""
from typing import List
from openai import OpenAI
from app.config import settings


_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Return a cached OpenAI client initialized with the configured API key.

    Returns:
        OpenAI: A singleton-like client instance reused across calls.
    """
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts using the configured OpenAI embedding model.

    Args:
        texts: List of input strings to embed.

    Returns:
        List[List[float]]: One embedding vector per input text.
    """
    if not texts:
        return []
    client = get_client()
    # OpenAI batches internally up to certain limits; we send one batch request here.
    resp = client.embeddings.create(model=settings.OPENAI_EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def embed_query(text: str) -> List[float]:
    """Embed a single query string and return its embedding vector.

    Args:
        text: The query to embed.

    Returns:
        List[float]: The embedding vector for the query.
    """
    client = get_client()
    resp = client.embeddings.create(model=settings.OPENAI_EMBEDDING_MODEL, input=[text])
    return resp.data[0].embedding
