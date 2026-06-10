"""Central configuration, loaded from `.env`.

Every module reads settings from here — no module reads os.environ directly.
Only the Phase 1 settings are surfaced; other keys in .env are ignored for now.
"""

from __future__ import annotations

from functools import lru_cache

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

    # ---------- Local inference (Ollama) ----------
    ollama_base_url: str = "http://localhost:11434"
    local_model: str = "qwen2.5:32b"
    embedding_model: str = "bge-m3"
    embedding_dim: int = 1024

    # ---------- Frontier API model (Phase 4 "hard query" route) ----------
    # Talks to an Anthropic-compatible endpoint. anthropic_base_url can point at a
    # gateway (corp proxy / vendor); leave blank to hit the public Anthropic API.
    anthropic_api_key: str = ""
    anthropic_base_url: str = ""
    api_model: str = "claude-sonnet-4-6"  # see routing.pricing for the cost table

    # ---------- Router (Phase 4) ----------
    router_strategy: str = "heuristic"  # heuristic | classifier
    router_hard_threshold: float = 0.7  # difficulty >= this routes to the API model

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
