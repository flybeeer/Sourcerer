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

    # ---------- Retrieval ----------
    chunk_size: int = 512  # approx words per chunk (see ingestion.chunking)
    chunk_overlap: int = 64
    top_k_vector: int = 20  # candidates pulled from vector search
    top_k_final: int = 5  # chunks actually passed to the model

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
