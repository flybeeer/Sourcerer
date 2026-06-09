"""Vector-only retrieval over pgvector (Phase 1).

Cosine similarity via the `<=>` distance operator. Phase 2 adds BM25 + fusion +
reranking alongside this; for now this is the whole retrieval story.
"""

from __future__ import annotations

from dataclasses import dataclass

from sourcerer.db.session import connect
from sourcerer.ingestion.embeddings import embed_query


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with its similarity score."""

    id: int
    source: str
    chunk_index: int
    content: str
    score: float  # cosine similarity in [−1, 1]; higher is closer


def search(query: str, top_k: int) -> list[RetrievedChunk]:
    """Embed the query and return the top_k most similar chunks."""
    query_embedding = embed_query(query)
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT id, source, chunk_index, content,
                   1 - (embedding <=> %s) AS score
            FROM chunks
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (query_embedding, query_embedding, top_k),
        ).fetchall()

    return [
        RetrievedChunk(id=row[0], source=row[1], chunk_index=row[2], content=row[3], score=row[4])
        for row in rows
    ]
