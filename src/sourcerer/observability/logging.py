"""Query logging — persist one row per answered query, and read recent history.

Phase 1 records the essentials (query, answer, citation count, latency). Later
phases extend this with the retrieval trace, route taken, tokens, and cost.
"""

from __future__ import annotations

from sourcerer.db.session import connect


def log_query(query: str, answer: str, num_citations: int, latency_ms: int) -> None:
    """Insert a record of an answered query."""
    with connect() as conn:
        conn.execute(
            "INSERT INTO query_log (query, answer, num_citations, latency_ms) "
            "VALUES (%s, %s, %s, %s)",
            (query, answer, num_citations, latency_ms),
        )


def recent_queries(limit: int = 20) -> list[dict]:
    """Return the most recent queries, newest first."""
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT id, query, answer, num_citations, latency_ms, created_at
            FROM query_log
            ORDER BY created_at DESC, id DESC
            LIMIT %s
            """,
            (limit,),
        ).fetchall()
    return [
        {
            "id": r[0],
            "query": r[1],
            "answer": r[2],
            "num_citations": r[3],
            "latency_ms": r[4],
            "created_at": r[5].isoformat(),
        }
        for r in rows
    ]
