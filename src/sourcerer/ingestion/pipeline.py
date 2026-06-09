"""Ingestion pipeline: load -> chunk -> embed -> store in pgvector.

Re-ingesting the same source is idempotent: existing rows for that source are
deleted first, so editing a document and re-running replaces its chunks.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sourcerer.config import get_settings
from sourcerer.db.session import connect, init_schema, to_vector_literal
from sourcerer.ingestion.chunking import chunk
from sourcerer.ingestion.embeddings import embed_texts
from sourcerer.ingestion.loaders import iter_documents

logger = logging.getLogger(__name__)


def ingest_directory(root: Path) -> dict[str, int]:
    """Ingest every supported document under `root`.

    Returns a small summary: {"documents": n, "chunks": m}.
    """
    settings = get_settings()
    init_schema()

    total_docs = 0
    total_chunks = 0

    with connect() as conn:
        for path, text in iter_documents(root):
            source = path.name
            chunks = chunk(text, settings)
            if not chunks:
                logger.warning("No chunks produced for %s; skipping", source)
                continue

            embeddings = embed_texts(chunks)

            # Replace any prior version of this source.
            conn.execute("DELETE FROM chunks WHERE source = %s", (source,))
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (source, chunk_index, content, embedding) "
                    "VALUES (%s, %s, %s, %s::vector)",
                    [
                        (source, i, content, to_vector_literal(embedding))
                        for i, (content, embedding) in enumerate(
                            zip(chunks, embeddings, strict=True)
                        )
                    ],
                )
            total_docs += 1
            total_chunks += len(chunks)
            logger.info("Ingested %s (%d chunks)", source, len(chunks))

    return {"documents": total_docs, "chunks": total_chunks}
