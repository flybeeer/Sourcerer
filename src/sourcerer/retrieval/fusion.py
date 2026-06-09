"""Reciprocal Rank Fusion (RRF).

Combines several ranked result lists into one. Each chunk's fused score is the
sum over the lists of 1 / (k + rank), where rank is 1-based within a list. Only
ranks matter, so lists with incomparable score scales (cosine vs ts_rank) fuse
cleanly. Chunks are de-duplicated by id.

Reference: Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet..."
"""

from __future__ import annotations

from dataclasses import replace

from sourcerer.retrieval.types import RetrievedChunk


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]], k: int
) -> list[RetrievedChunk]:
    """Fuse ranked lists into one list ordered by descending RRF score."""
    scores: dict[int, float] = {}
    first_seen: dict[int, RetrievedChunk] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(chunk.id, chunk)

    ordered_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    # Carry the RRF score onto the returned chunk for transparency/debugging.
    return [replace(first_seen[cid], score=scores[cid]) for cid in ordered_ids]
