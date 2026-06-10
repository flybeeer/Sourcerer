"""Query logging — persist one row per answered query, and read it back.

Phase 1 recorded the essentials (query, answer, citation count, latency). Phase 4
adds the routing trace: which route was taken, the model, token usage, estimated
cost, and the router's rationale — so the route is auditable per query and the
cost report can aggregate over the log.
"""

from __future__ import annotations

from sourcerer.db.session import connect


def log_query(
    query: str,
    answer: str,
    num_citations: int,
    latency_ms: int,
    *,
    route: str | None = None,
    model: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float = 0.0,
    router_reason: str | None = None,
    difficulty: float | None = None,
) -> None:
    """Insert a record of an answered query, including the Phase 4 routing trace."""
    with connect() as conn:
        conn.execute(
            "INSERT INTO query_log "
            "(query, answer, num_citations, latency_ms, route, model, "
            " input_tokens, output_tokens, cost_usd, router_reason, difficulty) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                query,
                answer,
                num_citations,
                latency_ms,
                route,
                model,
                input_tokens,
                output_tokens,
                cost_usd,
                router_reason,
                difficulty,
            ),
        )


def recent_queries(limit: int = 20) -> list[dict]:
    """Return the most recent queries, newest first (with the routing trace)."""
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT id, query, answer, num_citations, latency_ms, created_at,
                   route, model, input_tokens, output_tokens, cost_usd,
                   router_reason, difficulty
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
            "route": r[6],
            "model": r[7],
            "input_tokens": r[8],
            "output_tokens": r[9],
            "cost_usd": r[10],
            "router_reason": r[11],
            "difficulty": r[12],
        }
        for r in rows
    ]


def routed_queries(limit: int = 1000) -> list[dict]:
    """Return rows that carry a routing trace (Phase 4+), newest first.

    Used by the cost report; pre-Phase-4 rows (route IS NULL) are excluded.
    """
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT route, model, input_tokens, output_tokens, cost_usd, latency_ms
            FROM query_log
            WHERE route IS NOT NULL
            ORDER BY created_at DESC, id DESC
            LIMIT %s
            """,
            (limit,),
        ).fetchall()
    return [
        {
            "route": r[0],
            "model": r[1],
            "input_tokens": r[2],
            "output_tokens": r[3],
            "cost_usd": r[4],
            "latency_ms": r[5],
        }
        for r in rows
    ]
