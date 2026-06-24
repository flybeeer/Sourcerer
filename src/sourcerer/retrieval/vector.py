"""Vector retrieval over pgvector.

Cosine similarity via the `<=>` distance operator. Used both as the Phase 1
vector-only path and as one input to the Phase 2 hybrid pipeline.
"""

from __future__ import annotations

from sourcerer.db.session import connect, to_vector_literal
from sourcerer.ingestion.embeddings import embed_query
from sourcerer.retrieval.filter import source_clause
from sourcerer.retrieval.types import RetrievedChunk


def search(
    query: str,
    top_k: int,
    allowed_sources: set[str] | None = None,
    reader_principal: str | None = None,
) -> list[RetrievedChunk]:
    """Embed the query and return the top_k most similar chunks.

    `allowed_sources` (Phase 10b governance gate) pushes a `source = ANY(...)`
    predicate down so forbidden chunks never enter the candidate set. None = no
    filter; an empty set yields no rows. `reader_principal` (RLS backstop) runs the
    query under the restricted reader role so the DB filters rows for it too.
    """
    query_embedding = to_vector_literal(embed_query(query))
    clause, extra = source_clause(allowed_sources)
    with connect(reader_principal=reader_principal) as conn:
        rows = conn.execute(
            f"""
            SELECT id, source, chunk_index, content,
                   1 - (embedding <=> %s::vector) AS score
            FROM chunks
            WHERE TRUE{clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, *extra, query_embedding, top_k),
        ).fetchall()

    return [
        RetrievedChunk(id=row[0], source=row[1], chunk_index=row[2], content=row[3], score=row[4])
        for row in rows
    ]
