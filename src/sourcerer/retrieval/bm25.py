"""BM25-style keyword retrieval via Postgres full-text search.

Uses the stored `content_tsv` column (see db.session.init_schema) with
`ts_rank_cd` for ranking. Postgres FTS ranking is not literally BM25, but it
provides the keyword-match ordering the hybrid pipeline needs — and for RRF only
the rank order matters, not the absolute scores.

We OR the query terms together (rather than plainto_tsquery's default AND) so a
chunk matching *any* keyword is a candidate, ranked by how well it matches. This
is the right recall behaviour for keyword search, especially since the 'simple'
config keeps stopwords. Note: 'simple' does not stem or segment languages without
whitespace (e.g. Thai) well — good enough for Phase 2; revisit if needed.
"""

from __future__ import annotations

from sourcerer.db.session import connect
from sourcerer.retrieval.types import RetrievedChunk

# plainto_tsquery renders as "'a' & 'b' & 'c'"; turn the ANDs into ORs.
_OR_QUERY = "replace(plainto_tsquery('simple', %s)::text, ' & ', ' | ')::tsquery"


def search(query: str, top_k: int) -> list[RetrievedChunk]:
    """Return the top_k chunks matching any query keyword, best first."""
    with connect() as conn:
        rows = conn.execute(
            f"""
            WITH q AS (SELECT {_OR_QUERY} AS query)
            SELECT id, source, chunk_index, content,
                   ts_rank_cd(content_tsv, q.query) AS score
            FROM chunks, q
            WHERE content_tsv @@ q.query
            ORDER BY score DESC
            LIMIT %s
            """,
            (query, top_k),
        ).fetchall()

    return [
        RetrievedChunk(id=row[0], source=row[1], chunk_index=row[2], content=row[3], score=row[4])
        for row in rows
    ]
