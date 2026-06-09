"""Postgres connections + schema management for pgvector.

Phase 1 keeps this deliberately simple: open a connection when needed, register
the pgvector type adapters, and expose an idempotent schema initializer. A
connection pool can come later if/when concurrency matters.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector

from sourcerer.config import get_settings


def to_vector_literal(vec: list[float]) -> str:
    """Format an embedding as a pgvector text literal, e.g. "[0.1,0.2,0.3]".

    Bound as a `%s::vector` parameter so Postgres parses it as a `vector` (a plain
    Python list would otherwise be sent as double precision[], which the `<=>`
    operator does not accept).
    """
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


@contextmanager
def connect() -> Iterator[psycopg.Connection]:
    """Yield a pgvector-aware connection, committing on success.

    Ensures the `vector` extension is present (harmless if already enabled) and
    registers the vector type adapters so Python lists map to/from `vector`.
    """
    settings = get_settings()
    with psycopg.connect(settings.dsn) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        yield conn


def init_schema() -> None:
    """Create the chunks table + vector index if they don't exist.

    The embedding dimension comes from config so it always matches the model.
    """
    settings = get_settings()
    dim = settings.embedding_dim
    with connect() as conn:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id          BIGSERIAL PRIMARY KEY,
                source      TEXT        NOT NULL,
                chunk_index INTEGER     NOT NULL,
                content     TEXT        NOT NULL,
                embedding   vector({dim}) NOT NULL,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """)
        # HNSW index for cosine similarity search. Fine for a small corpus;
        # revisit parameters when the corpus grows.
        conn.execute("""
            CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
            ON chunks USING hnsw (embedding vector_cosine_ops)
            """)
        # Query log — one row per answered query (the start of observability).
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_log (
                id              BIGSERIAL PRIMARY KEY,
                query           TEXT        NOT NULL,
                answer          TEXT        NOT NULL,
                num_citations   INTEGER     NOT NULL,
                latency_ms      INTEGER     NOT NULL,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """)
