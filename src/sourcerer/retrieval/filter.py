"""Source-level predicate pushdown for the governance gate (Phase 10b).

The gate resolves which sources a principal may read; we push that set into the
retrieval SQL as `AND source = ANY(:allowed)` so forbidden chunks are excluded
in the database, before fusion/rerank — the same predicate-pushdown idea as the
Phase 7 SQL work, applied to document retrieval.
"""

from __future__ import annotations


def source_clause(allowed_sources: set[str] | None) -> tuple[str, tuple]:
    """Build the optional source filter as a SQL fragment + bound params.

    Returns ("", ()) when there is no filter (governance off), else
    (" AND source = ANY(%s)", ([..sources..],)). An empty allowed set is a real
    filter that matches nothing — never "no filter".
    """
    if allowed_sources is None:
        return "", ()
    return " AND source = ANY(%s)", (list(allowed_sources),)
