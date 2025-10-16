"""Application package containing the API, configuration, data access, retrieval/generation
pipelines, and supporting utilities.

Submodules overview:
- main: FastAPI application bootstrap and lifecycle.
- router: API route definitions, dependencies, and handlers.
- config: Application settings and environment variable loading.
- db: Database engine/session management helpers.
- models: ORM models and relationships.
- schemas: Pydantic request/response models for API contracts.
- retrieval: Document retrieval pipeline (indexes, queries, scoring).
- generation: LLM generation helpers (prompting, parameters, outputs).
- embedding: Embedding utilities and providers.
- reranker: Reranking components for retrieval results.
- cache: Caching utilities (keys, TTLs, invalidation).
- ingestion: Offline/ETL ingestion jobs (e.g., Stripe ingest).
- evals: Evaluation scripts and helpers (e.g., RAGAS runner).
- obs: Observability utilities (tracing/spans).
- utils: General-purpose helper functions.
"""
