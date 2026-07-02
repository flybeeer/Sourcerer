"""Ingestion pipeline: load -> chunk -> embed -> store in pgvector.

Three sources feed the same store: files under a directory (`ingest_directory`),
rows from a SQLite database (`ingest_sql`, Phase 7), and Confluence pages
(`ingest_confluence`). All share `_store_document`, so retrieval/citations don't
care where a chunk came from.

Re-ingesting the same source is idempotent: existing rows for that source are
deleted first, so editing a document (or row) and re-running replaces its chunks.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path

import psycopg

from sourcerer.config import Settings, get_settings
from sourcerer.db.session import connect, init_schema, to_vector_literal
from sourcerer.ingestion.chunking import chunk
from sourcerer.ingestion.confluence import iter_pages
from sourcerer.ingestion.embeddings import embed_texts
from sourcerer.ingestion.loaders import iter_documents
from sourcerer.sqlkb.loader import iter_rows

logger = logging.getLogger(__name__)


def _store_document(conn: psycopg.Connection, source: str, text: str, settings: Settings) -> int:
    """Chunk, embed, and (idempotently) replace one document's rows. Returns chunk count.

    Replaces any prior version of `source`, so re-ingesting is safe. Returns 0 when
    the text produces no chunks (caller decides whether to count the document).
    """
    chunks = chunk(text, settings)
    if not chunks:
        logger.warning("No chunks produced for %s; skipping", source)
        return 0

    embeddings = embed_texts(chunks)
    conn.execute("DELETE FROM chunks WHERE source = %s", (source,))
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (source, chunk_index, content, embedding) "
            "VALUES (%s, %s, %s, %s::vector)",
            [
                (source, i, content, to_vector_literal(embedding))
                for i, (content, embedding) in enumerate(zip(chunks, embeddings, strict=True))
            ],
        )
    logger.info("Ingested %s (%d chunks)", source, len(chunks))
    return len(chunks)


def _store_metadata(conn: psycopg.Connection, source: str, metadata: dict) -> None:
    """Upsert a document's metadata, keyed by its `source` label."""
    conn.execute(
        """
        INSERT INTO document_metadata (source, metadata, updated_at)
        VALUES (%s, %s::jsonb, now())
        ON CONFLICT (source) DO UPDATE SET metadata = EXCLUDED.metadata, updated_at = now()
        """,
        (source, json.dumps(metadata)),
    )


def _ingest(items: Iterable[tuple[str, str]]) -> dict:
    """Ingest an iterable of (source, text) pairs.

    Returns {documents, chunks, sources} — `sources` lists the names actually
    stored (drops empty docs), so callers like the governance tagger know exactly
    which assets to catalogue.
    """
    settings = get_settings()
    init_schema()

    sources: list[str] = []
    total_chunks = 0
    with connect() as conn:
        for source, text in items:
            n = _store_document(conn, source, text, settings)
            if n:
                sources.append(source)
                total_chunks += n
    return {"documents": len(sources), "chunks": total_chunks, "sources": sources}


def ingest_directory(root: Path) -> dict:
    """Ingest every supported document under `root`. Returns {documents, chunks, sources}."""
    return _ingest((path.name, text) for path, text in iter_documents(root))


def ingest_sql(
    db_path: str | Path,
    query: str,
    *,
    id_col: str | None = None,
    source_prefix: str | None = None,
) -> dict:
    """Ingest rows from a SQLite database as documents (Phase 7, Path A).

    Each row returned by `query` becomes one document; `id_col` (if given) names
    its source for stable, traceable citations. Returns {documents, chunks, sources}.
    """
    return _ingest(
        iter_rows(db_path, query, id_col=id_col, source_prefix=source_prefix)
    )


def ingest_confluence(
    cql: str,
    *,
    base_url: str,
    email: str,
    api_token: str,
    max_pages: int = 1000,
) -> dict:
    """Ingest Confluence pages matching a CQL query as documents.

    Each page becomes one document (1 page = 1 document, like ingest_sql's 1 row
    = 1 document); source label is `confluence:<page_id>`. Returns {documents,
    chunks, sources}.

    Unlike `ingest_directory`/`ingest_sql`, this doesn't go through `_ingest`:
    Confluence pages carry real per-document metadata (space, author, labels,
    URL, ...) that the other two sources don't have, so it needs its own loop
    to store it alongside the chunks.
    """
    settings = get_settings()
    init_schema()

    sources: list[str] = []
    total_chunks = 0
    with connect() as conn:
        for source, text, metadata in iter_pages(
            cql, base_url=base_url, email=email, api_token=api_token, max_pages=max_pages
        ):
            n = _store_document(conn, source, text, settings)
            if n:
                sources.append(source)
                total_chunks += n
                _store_metadata(conn, source, metadata)
    return {"documents": len(sources), "chunks": total_chunks, "sources": sources}
