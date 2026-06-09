"""Shared retrieval types.

`RetrievedChunk` is produced by every retrieval path (vector, BM25, fusion,
rerank). The meaning of `score` depends on the producer (cosine similarity,
ts_rank, RRF score, or rerank score) — it is always "higher is better".
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    id: int
    source: str
    chunk_index: int
    content: str
    score: float  # higher = more relevant (semantics depend on the producer)
