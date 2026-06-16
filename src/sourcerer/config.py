"""Central configuration, loaded from `.env`.

Every module reads settings from here — no module reads os.environ directly.
Only the Phase 1 settings are surfaced; other keys in .env are ignored for now.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # .env has keys for later phases; ignore them here
    )

    # ---------- App ----------
    app_env: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ---------- Database (Postgres + pgvector) ----------
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "sourcerer"
    postgres_user: str = "sourcerer"
    postgres_password: str = "change_me"

    # ---------- Local inference ----------
    # Generation backend for the local route: ollama (dev) or vllm (prod).
    # Embeddings always run on Ollama (bge-m3) regardless of this.
    local_backend: str = "ollama"  # ollama | vllm
    ollama_base_url: str = "http://localhost:11434"
    local_model: str = "qwen2.5:32b"
    embedding_model: str = "bge-m3"
    embedding_dim: int = 1024

    # vLLM (Phase 5) — OpenAI-compatible server for higher-throughput local serving.
    vllm_base_url: str = ""  # e.g. http://localhost:8001
    vllm_model: str = ""  # the model id vLLM was launched with

    # ---------- Frontier API model (Phase 4 "hard query" route) ----------
    # Talks to an Anthropic-compatible endpoint. anthropic_base_url can point at a
    # gateway (corp proxy / vendor); leave blank to hit the public Anthropic API.
    anthropic_api_key: str = ""
    anthropic_base_url: str = ""
    api_model: str = "claude-sonnet-4-6"  # see routing.pricing for the cost table

    # ---------- Router (Phase 4) ----------
    router_strategy: str = "heuristic"  # heuristic | classifier
    router_hard_threshold: float = 0.7  # difficulty >= this routes to the API model

    # ---------- Guardrails (Phase 5) ----------
    enable_injection_check: bool = True  # reject obvious prompt-injection inputs
    # Optional relevance floor: when a cross-encoder reranker ran and the top
    # chunk scores below this, treat retrieval as "no relevant context" → I don't
    # know. None = disabled (rely on the empty-context guardrail only).
    min_relevance_score: float | None = None

    @field_validator("min_relevance_score", mode="before")
    @classmethod
    def _blank_to_none(cls, v):
        """Treat a blank env var (e.g. MIN_RELEVANCE_SCORE=) as unset (None)."""
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    # ---------- Chunking ----------
    chunk_strategy: str = "fixed"  # fixed | semantic
    chunk_size: int = 512  # approx words per chunk (see ingestion.chunking)
    chunk_overlap: int = 64  # used by the fixed strategy
    semantic_threshold: float = 0.5  # similarity break point for the semantic strategy

    # ---------- Retrieval ----------
    retrieval_mode: str = "hybrid"  # hybrid | vector  (Phase 1 path kept behind this flag)
    top_k_vector: int = 20  # candidates pulled from vector search
    top_k_bm25: int = 20  # candidates pulled from BM25/keyword search
    top_k_final: int = 5  # chunks actually passed to the model
    rrf_k: int = 60  # reciprocal rank fusion constant

    # ---------- Reranker ----------
    reranker_type: str = "none"  # none | local | cohere | llm  (none = keep fused order)
    reranker_model: str = "bge-reranker-v2-m3"
    cohere_api_key: str = ""

    # ---------- Evaluation (Phase 3) ----------
    eval_set_path: str = "eval/eval_set.jsonl"
    eval_corpus_path: str = "eval/corpus"
    eval_judge_model: str = "qwen2.5:7b"  # LLM-as-judge; swap for a frontier model later

    # ---------- GraphRAG (Phase 6, optional) ----------
    graphrag_enabled: bool = False  # parallel graph-retrieval path off by default
    graphrag_extraction_model: str = "qwen2.5:32b"  # LOCAL model for extraction (cost control)
    # Use the frontier API model for graph extraction + summaries instead of the
    # local model. Off by default (the blueprint says keep extraction local for
    # cost); turn on only for content a small local model can't extract — e.g. it
    # won't emit the structured format for Thai/garbled OCR text. Costs API tokens
    # per chunk; keep the corpus tiny. Needs ANTHROPIC_API_KEY.
    graphrag_extract_with_api: bool = False
    graphrag_root: str = "./graphrag"  # where graph artifacts are written/read (json store)
    graphrag_overview_eval_set: str = "eval/overview_eval_set.jsonl"
    # Where the graph is stored. json = a single file loaded per query (simple, dev,
    # small corpora). postgres = entities/relationships/communities in Postgres with
    # community summaries embedded in pgvector, so global search ranks communities by
    # similarity to the query (not top-N by size) — the production path.
    graphrag_store: str = "json"  # json | postgres
    # Send the final global-search *reduce* (synthesis) to the frontier API model
    # while the many cheap map calls stay local — Phase 4 routing, inside GraphRAG.
    # Needs ANTHROPIC_API_KEY; map + indexing remain local regardless.
    graphrag_reduce_with_api: bool = False
    # After a source is deleted, rebuild the graph in the background: drop
    # communities whose source files are all gone and trim deleted files from the
    # rest (keyed on stable source names, not rotating chunk ids). Cheap — no
    # re-extraction, no LLM — and removes the deleted doc's themes for good.
    graphrag_reindex_on_delete: bool = False

    # ---------- SQL knowledge base (Phase 7, optional) ----------
    # Treat a SQLite database as a knowledge source. Two paths share this flag:
    #   - ingestion: rows pulled via SELECT become RAG documents (scripts/ingest_sql.py);
    #   - Text-to-SQL: analytical/aggregation questions ("total sales last year") are
    #     answered by generating a read-only SELECT and running it on the SQLite file.
    sql_kb_enabled: bool = False
    sql_kb_path: str = "./data/kb.sqlite"  # the source DB file (opened read-only)
    # Storage engine behind the source DB: sqlite (demo) or duckdb (columnar; for
    # analytics-scale tables, aggregations push down to a column store).
    sql_kb_backend: str = "sqlite"  # sqlite | duckdb
    sql_kb_max_rows: int = 50  # cap rows a generated query may return (safety + cost)
    # Runtime cost guards on a generated query (defence vs a full-scan on a big table):
    #   timeout  — wall-clock cap; aborts the query on both backends (0 = off).
    #   max_scan_ops — SQLite VM-op budget, a proxy for rows/bytes scanned (0 = off);
    #     the bytes-scanned analogue on a real warehouse (e.g. BigQuery maximum_bytes_billed).
    sql_kb_timeout_s: float = 5.0
    sql_kb_max_scan_ops: int = 0
    # Schema retrieval: when a source has more tables than this, send only the most
    # relevant ones to the Text-to-SQL prompt (ranked lexical + embedding → RRF,
    # same hybrid idea as document retrieval) instead of dumping every table's DDL —
    # which would blow the context and degrade SQL on a wide schema. 0 = always all.
    sql_kb_schema_top_k: int = 8
    # Generate the SQL with the frontier API model instead of the local model. Off by
    # default (local keeps it free/private); turn on for harder schemas. Needs a key.
    sql_kb_generate_with_api: bool = False

    @property
    def dsn(self) -> str:
        """Build the Postgres connection string from parts.

        Composed from the individual POSTGRES_* values rather than reading
        DATABASE_URL, so container overrides (e.g. POSTGRES_HOST=db in
        docker-compose) work cleanly. DATABASE_URL in .env is for external tools.
        """
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (read .env once per process)."""
    return Settings()
