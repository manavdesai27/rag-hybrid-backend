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
    vector_available = False
    with engine.connect() as conn:
        try:
            # Check if pgvector is available; attempt to enable if missing
            res = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'"))
            if res.scalar() == 1:
                vector_available = True
            else:
                try:
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    vector_available = True
                except Exception as e:
                    # Tables using the 'vector' type will fail without the extension
                    raise RuntimeError(
                        "pgvector extension is not available on the configured database. "
                        "Enable it (CREATE EXTENSION vector) or point DATABASE_URL to a Postgres "
                        "instance with pgvector installed."
                    ) from e
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # Import models after Base is defined
    from app import models  # noqa: F401

    # Create tables
    Base.metadata.create_all(bind=engine)

    # Create vector index (ivfflat) for embeddings if not exists
    # Note: Requires pgvector >= 0.4.0; table/index names must match models.
    if vector_available:
        with engine.connect() as conn:
            try:
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
            except Exception:
                conn.rollback()
                # Non-fatal: index creation can be done later once data exists


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
