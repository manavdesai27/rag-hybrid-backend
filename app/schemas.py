"""Pydantic request/response schemas for the API.

Defines the public contracts used by the FastAPI endpoints:
- AskRequest: Input payload for the question-answering endpoint.
- Citation: Evidence snippet reference returned alongside answers.
- AskResponse: Output payload with the final answer and citations.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request body for asking a question to the RAG pipeline.

    Attributes:
        question: The user question to answer.
        max_tokens: Optional cap on the number of tokens for the generated answer.
    """
    question: str = Field(..., min_length=1, description="User question")
    max_tokens: Optional[int] = Field(
        default=None,
        ge=64,
        le=8192,
        description="Desired maximum tokens for the answer (overrides server default)",
    )


class Citation(BaseModel):
    """A reference to a supporting snippet for an answer.

    Attributes:
        source: Optional title or section name for display.
        url: The origin URL of the snippet.
        snippet: The extracted text snippet shown as evidence.
    """
    source: Optional[str] = None  # title/section
    url: str
    snippet: str


class AskResponse(BaseModel):
    """Response body returned by the RAG pipeline.

    Attributes:
        answer: The generated answer text.
        citations: List of supporting citations used to form the answer.
        latency_ms: End-to-end latency for the request in milliseconds.
        used_cache: Whether the answer was served from cache.
    """
    answer: str
    citations: List[Citation]
    latency_ms: int
    used_cache: bool = False
