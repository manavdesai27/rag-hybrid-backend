"""Database ORM models.

Defines persistent entities used by the retrieval pipeline. Currently includes:
- Chunk: a semantically searchable content chunk with metadata and a pgvector
  embedding used for vector similarity search. Common indexes exist on doc_id and url.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Index
from pgvector.sqlalchemy import Vector

from app.db import Base
from app.config import settings


class Chunk(Base):
    """Vector-embedded document chunk used for retrieval.

    Each row represents a chunk of source content along with:
    - document-level metadata (doc_id, url, title, section)
    - chunk-level metadata (position, content, tokens)
    - an embedding vector (pgvector) for ANN search

    Indexes:
        - idx_chunks_doc: speeds up filtering/grouping by document
        - idx_chunks_url: speeds up lookups by URL

    Notes:
        The embedding dimension is derived from settings.EMBEDDING_DIM and should
        match the embedding model configured in app.config.Settings.
    """
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # Document/page level metadata
    doc_id = Column(String(64), nullable=False)         # stable hash of URL/path
    url = Column(String(1024), nullable=False)
    title = Column(String(512), nullable=True)
    section = Column(String(512), nullable=True)

    # Chunk-level metadata
    position = Column(Integer, nullable=False, default=0)  # order within a doc
    content = Column(Text, nullable=False)
    tokens = Column(Integer, nullable=True)

    # Embedding vector
    embedding = Column(Vector(dim=settings.EMBEDDING_DIM), nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_chunks_doc", "doc_id"),
        Index("idx_chunks_url", "url"),
    )
