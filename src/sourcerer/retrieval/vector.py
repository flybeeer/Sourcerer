"""Vector retrieval over pgvector.

Cosine similarity via the `<=>` distance operator. Used both as the Phase 1
vector-only path and as one input to the Phase 2 hybrid pipeline.
"""

from __future__ import annotations

from sourcerer.db.session import connect, to_vector_literal
from sourcerer.ingestion.embeddings import embed_query
from sourcerer.retrieval.types import RetrievedChunk


def search(query: str, top_k: int) -> list[RetrievedChunk]:
    """Embed the query and return the top_k most similar chunks."""
    query_embedding = to_vector_literal(embed_query(query))
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT id, source, chunk_index, content,
                   1 - (embedding <=> %s::vector) AS score
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, query_embedding, top_k),
        ).fetchall()

    return [
        RetrievedChunk(id=row[0], source=row[1], chunk_index=row[2], content=row[3], score=row[4])
        for row in rows
    ]
