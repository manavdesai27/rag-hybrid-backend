"""Application configuration and environment-driven settings.

Defines the Settings class based on pydantic-settings to centralize configuration for:
- API keys and model names
- Data stores (PostgreSQL, Redis) and cache defaults
- Ingestion parameters (seed URLs, chunking)
- Retrieval/generation knobs
- Optional observability (Langfuse)
- Evals thresholds and dataset

A light-weight local safety warning is printed if OPENAI_API_KEY is not set when not running in Docker.
"""
import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings loaded from environment variables.

    Uses pydantic-settings to populate fields from a .env file or process env.
    See individual field names for semantics and safe defaults.
    """
    # Required
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")

    # Models
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"  # 1536 dims

    # Data stores
    DATABASE_URL: str = "postgresql+psycopg2://rag_user:rag_pass@db:5432/rag_db"
    REDIS_URL: str = "redis://redis:6379/0"
    CACHE_TTL_SECONDS: int = 600

    # Ingestion
    DOCS_SEED_URLS: str = "https://docs.stripe.com/api,https://docs.stripe.com/payments"
    MAX_DOCS: int = 30
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150

    # Retrieval/Generation
    TOP_K: int = 8
    RERANKER_ENABLED: bool = False
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-base"
    RERANKER_TOPN: int = 40
    REFUSAL_MIN_SIMILARITY: float = 0.72  # 0-1
    MAX_OUTPUT_TOKENS: int = 350

    # Observability (optional)
    LANGFUSE_HOST: str = ""
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""

    # Evals
    EVAL_API_BASE_URL: str = "http://localhost:8000"
    EVAL_DATASET_PATH: str = "data/evals/stripe_seed.jsonl"
    EVAL_MIN_FAITHFULNESS: float = 0.6
    EVAL_MIN_ANSWER_RELEVANCY: float = 0.6
    EVAL_MIN_CONTEXT_PRECISION: float = 0.4
    EVAL_MIN_CONTEXT_RECALL: float = 0.4

    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 20
    RATE_LIMIT_BURST_5S: int = 5
    RATE_LIMIT_PER_DAY: int = 500
    RATE_LIMIT_PER_QUESTION_INTERVAL_SECONDS: int = 5
    MAX_CONCURRENCY: int = 8
    PER_IP_MAX_CONCURRENCY: int = 2
    MAX_QUEUE_SIZE: int = 50
    TRUST_X_FORWARDED_FOR: bool = True
    REAL_IP_HEADER: str = "X-Forwarded-For"

    # Derived
    @property
    def EMBEDDING_DIM(self) -> int:
        """Embedding dimension for the configured embedding model.

        Returns:
            int: The vector dimension inferred from OPENAI_EMBEDDING_MODEL.
        """
        # Map common OpenAI embedding models to dimensions
        model = self.OPENAI_EMBEDDING_MODEL.lower()
        if "text-embedding-3-small" in model:
            return 1536
        if "text-embedding-3-large" in model:
            return 3072
        # Fallback
        return 1536

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


settings = Settings()

# Safety check for local dev (inside API container this must be set)
if os.environ.get("RUNNING_IN_DOCKER", "0") == "0":
    # Only warn in local context; container will require it
    if not settings.OPENAI_API_KEY:
        # Avoid raising to allow local scaffolding before setting .env
        print("[WARN] OPENAI_API_KEY not set. Set it in .env before running ingestion or /ask.")
