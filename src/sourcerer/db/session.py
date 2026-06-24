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
def connect(reader_principal: str | None = None) -> Iterator[psycopg.Connection]:
    """Yield a pgvector-aware connection, committing on success.

    Ensures the `vector` extension is present (harmless if already enabled) and
    registers the vector type adapters so Python lists map to/from `vector`.

    `reader_principal` (Phase 10d RLS backstop) connects under the restricted,
    non-superuser reader role instead and sets the `sourcerer.principal` session
    setting, so the row-level security policy on `chunks` filters rows in the
    database for that principal — independently of any app-level gate. The reader
    can't create extensions, so that step is skipped (the extension already exists).
    """
    settings = get_settings()
    if reader_principal is not None:
        with psycopg.connect(settings.reader_dsn) as conn:
            # set_config(name, value, is_local=false) → session-scoped GUC the RLS
            # policy reads via current_setting('sourcerer.principal', true).
            conn.execute("SELECT set_config('sourcerer.principal', %s, false)", (reader_principal,))
            register_vector(conn)
            yield conn
        return
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
        # Full-text search vector for BM25-style keyword retrieval (Phase 2).
        # Generated + stored so it stays in sync with content and backfills
        # existing rows. 'simple' config = language-agnostic tokenisation.
        conn.execute("""
            ALTER TABLE chunks
            ADD COLUMN IF NOT EXISTS content_tsv tsvector
            GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED
            """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS chunks_content_tsv_gin
            ON chunks USING gin (content_tsv)
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
        # Phase 4 routing instrumentation — added in place so existing logs keep
        # working (NULL route marks pre-Phase-4 rows; the report skips those).
        for ddl in (
            "route TEXT",
            "model TEXT",
            "input_tokens INTEGER NOT NULL DEFAULT 0",
            "output_tokens INTEGER NOT NULL DEFAULT 0",
            "cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0",
            "router_reason TEXT",
            "difficulty DOUBLE PRECISION",
            # Phase 10a — principal that issued the query (NULL = governance off).
            "principal TEXT",
            # Phase 10d — how many assets the gate hid from this principal (audit;
            # NULL = governance off / path doesn't count denials).
            "denied_assets INTEGER",
        ):
            conn.execute(f"ALTER TABLE query_log ADD COLUMN IF NOT EXISTS {ddl}")

        # GraphRAG postgres store (Phase 6 production path). Community summaries are
        # embedded so global search can rank communities by similarity to the query.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS graph_entities (
                id          BIGSERIAL PRIMARY KEY,
                name        TEXT    NOT NULL,
                type        TEXT    NOT NULL,
                description TEXT    NOT NULL,
                chunk_ids   INTEGER[] NOT NULL DEFAULT '{}',
                sources     TEXT[]    NOT NULL DEFAULT '{}'
            )
            """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS graph_relationships (
                id          BIGSERIAL PRIMARY KEY,
                source      TEXT NOT NULL,
                target      TEXT NOT NULL,
                description TEXT NOT NULL
            )
            """)
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS graph_communities (
                id                INTEGER PRIMARY KEY,
                summary           TEXT    NOT NULL,
                entity_names      TEXT[]  NOT NULL DEFAULT '{{}}',
                sources           TEXT[]  NOT NULL DEFAULT '{{}}',
                summary_embedding vector({dim})
            )
            """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS graph_communities_emb_hnsw
            ON graph_communities USING hnsw (summary_embedding vector_cosine_ops)
            """)
