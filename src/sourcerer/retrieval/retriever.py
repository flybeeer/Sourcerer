"""Retrieval entry point — selects vector-only or hybrid by mode.

The Phase 1 vector-only path is kept reachable via RETRIEVAL_MODE=vector (or a
per-request override) so Phase 3 can benchmark hybrid against it.
"""

from __future__ import annotations

from sourcerer.config import Settings
from sourcerer.retrieval import hybrid, vector
from sourcerer.retrieval.types import RetrievedChunk


def retrieve(
    query: str,
    settings: Settings,
    mode: str | None = None,
    top_k_final: int | None = None,
    allowed_sources: set[str] | None = None,
    reader_principal: str | None = None,
) -> list[RetrievedChunk]:
    """Retrieve chunks for a query.

    Args:
        mode: "vector" or "hybrid". Defaults to settings.retrieval_mode.
        top_k_final: number of chunks to return. Defaults to settings.top_k_final.
        allowed_sources: governance gate (Phase 10b) — restrict retrieval to these
            sources. None = no filter; an empty set returns nothing.
        reader_principal: RLS backstop (Phase 10d) — run under the restricted reader
            role so the database independently filters rows for this principal.
    """
    resolved_mode = (mode or settings.retrieval_mode).lower()
    final = top_k_final or settings.top_k_final

    if resolved_mode == "vector":
        return vector.search(query, final, allowed_sources, reader_principal)
    if resolved_mode == "hybrid":
        return hybrid.retrieve(
            query,
            settings,
            top_k_final=final,
            allowed_sources=allowed_sources,
            reader_principal=reader_principal,
        )
    raise ValueError(f"Unknown retrieval mode: {resolved_mode!r}")
