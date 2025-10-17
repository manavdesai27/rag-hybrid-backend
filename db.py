"""Database setup and session utilities for SQLAlchemy.

This module centralizes engine/session initialization, metadata base, and helpers:
- init_db: Ensures the pgvector extension exists and creates required tables and the
  IVFFLAT index over the chunks.embedding column for vector similarity search.
- session_scope: Context-managed transactional scope for imperative workflows.
- get_db: FastAPI dependency to yield a per-request SQLAlchemy Session.

Configuration is read from app.config.settings.DATABASE_URL.
"""
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

from app.config import settings

# SQLAlchemy setup
engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)
Base = declarative_base()


def init_db() -> None:
    """Initialize database extensions, tables, and vector indexes.

    Ensures pgvector extension is available, creates tables from SQLAlchemy metadata,
    and creates the IVFFLAT index over chunks.embedding if missing.

    This function is idempotent and safe to run multiple times.
    """
    with engine.connect() as conn:
        # Enable pgvector extension
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    # Import models after Base is defined
    from app import models  # noqa: F401

    # Create tables
    Base.metadata.create_all(bind=engine)

    # Create vector index (ivfflat) for embeddings if not exists
    # Note: Requires pgvector >= 0.4.0; table/index names must match models.
    with engine.connect() as conn:
        # Switch to IVF index (requires ANALYZE after populate for optimal perf)
        conn.execute(
            text(
                """
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_indexes WHERE indexname = 'idx_chunks_embedding_ivfflat'
                    ) THEN
                        CREATE INDEX idx_chunks_embedding_ivfflat
                        ON chunks USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100);
                    END IF;
                END$$;
                """
            )
        )
        # Lightweight BM25-style support could be added via tsvector GIN index; we keep BM25 in-memory for MVP.
        conn.commit()


@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations.

    Yields:
        Session: A SQLAlchemy session bound to the configured engine.

    Notes:
        - Commits on successful exit.
        - Rolls back and re-raises on exception.
        - Always closes the session at the end.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Generator:
    """FastAPI dependency that yields a SQLAlchemy Session.

    Yields:
        Session: A session tied to the current request lifecycle.

    Notes:
        Ensures the session is closed after the request finishes.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
