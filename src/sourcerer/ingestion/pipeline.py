"""Ingestion pipeline: load -> chunk -> embed -> store in pgvector.

Two sources feed the same store: files under a directory (`ingest_directory`) and
rows from a SQLite database (`ingest_sql`, Phase 7). Both share `_store_document`,
so retrieval/citations don't care where a chunk came from.

Re-ingesting the same source is idempotent: existing rows for that source are
deleted first, so editing a document (or row) and re-running replaces its chunks.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import psycopg

from sourcerer.config import Settings, get_settings
from sourcerer.db.session import connect, init_schema, to_vector_literal
from sourcerer.ingestion.chunking import chunk
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


def _ingest(items: Iterable[tuple[str, str]]) -> dict[str, int]:
    """Ingest an iterable of (source, text) pairs. Returns {documents, chunks}."""
    settings = get_settings()
    init_schema()

    total_docs = 0
    total_chunks = 0
    with connect() as conn:
        for source, text in items:
            n = _store_document(conn, source, text, settings)
            if n:
                total_docs += 1
                total_chunks += n
    return {"documents": total_docs, "chunks": total_chunks}


def ingest_directory(root: Path) -> dict[str, int]:
    """Ingest every supported document under `root`. Returns {documents, chunks}."""
    return _ingest((path.name, text) for path, text in iter_documents(root))


def ingest_sql(
    db_path: str | Path,
    query: str,
    *,
    id_col: str | None = None,
    source_prefix: str | None = None,
) -> dict[str, int]:
    """Ingest rows from a SQLite database as documents (Phase 7, Path A).

    Each row returned by `query` becomes one document; `id_col` (if given) names
    its source for stable, traceable citations. Returns {documents, chunks}.
    """
    return _ingest(
        iter_rows(db_path, query, id_col=id_col, source_prefix=source_prefix)
    )
