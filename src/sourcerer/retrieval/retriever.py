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
) -> list[RetrievedChunk]:
    """Retrieve chunks for a query.

    Args:
        mode: "vector" or "hybrid". Defaults to settings.retrieval_mode.
        top_k_final: number of chunks to return. Defaults to settings.top_k_final.
    """
    resolved_mode = (mode or settings.retrieval_mode).lower()
    final = top_k_final or settings.top_k_final

    if resolved_mode == "vector":
        return vector.search(query, final)
    if resolved_mode == "hybrid":
        return hybrid.retrieve(query, settings, top_k_final=final)
    raise ValueError(f"Unknown retrieval mode: {resolved_mode!r}")
