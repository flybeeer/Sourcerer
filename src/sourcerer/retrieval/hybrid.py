"""Hybrid retrieval pipeline: vector ∥ BM25 → RRF fusion → rerank.

query ──┬─► vector.search   (top_k_vector)  ─┐
        └─► bm25.search     (top_k_bm25)    ─┴─► RRF(rrf_k) ─► rerank ─► top_k_final
"""

from __future__ import annotations

from sourcerer.config import Settings
from sourcerer.retrieval import bm25, vector
from sourcerer.retrieval.fusion import reciprocal_rank_fusion
from sourcerer.retrieval.rerank import get_reranker
from sourcerer.retrieval.types import RetrievedChunk


def retrieve(query: str, settings: Settings, top_k_final: int) -> list[RetrievedChunk]:
    """Run the full hybrid pipeline and return the final top_k_final chunks."""
    vector_hits = vector.search(query, settings.top_k_vector)
    bm25_hits = bm25.search(query, settings.top_k_bm25)

    fused = reciprocal_rank_fusion([vector_hits, bm25_hits], k=settings.rrf_k)

    reranker = get_reranker(settings)
    return reranker.rerank(query, fused, top_k_final)
