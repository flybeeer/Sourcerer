"""Retrieval metrics: recall@k, MRR, hit rate.

Evaluated at the document level: a retrieved chunk counts as relevant if its
`source` is in the question's `relevant_doc_ids`. `retrieved_sources` is the
ordered list of chunk sources returned by a retrieval config (rank order, may
contain duplicates).
"""

from __future__ import annotations


def recall_at_k(retrieved_sources: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant docs that appear in the top-k retrieved chunks."""
    if not relevant:
        return 0.0
    top = set(retrieved_sources[:k])
    return len(top & relevant) / len(relevant)


def reciprocal_rank(retrieved_sources: list[str], relevant: set[str], k: int) -> float:
    """Reciprocal rank of the first relevant doc within the top-k (0 if none)."""
    for rank, source in enumerate(retrieved_sources[:k], start=1):
        if source in relevant:
            return 1.0 / rank
    return 0.0


def hit(retrieved_sources: list[str], relevant: set[str], k: int) -> float:
    """1.0 if at least one relevant doc is in the top-k, else 0.0."""
    return 1.0 if set(retrieved_sources[:k]) & relevant else 0.0


def per_query_metrics(retrieved_sources: list[str], relevant_doc_ids: list[str], k: int) -> dict:
    """Compute recall@k, reciprocal rank, and hit for one query."""
    relevant = set(relevant_doc_ids)
    return {
        "has_labels": bool(relevant),
        "recall@k": recall_at_k(retrieved_sources, relevant, k),
        "mrr": reciprocal_rank(retrieved_sources, relevant, k),
        "hit": hit(retrieved_sources, relevant, k),
    }


def aggregate(per_query: list[dict]) -> dict:
    """Average a list of per-query metric dicts (ignoring queries with no labels)."""
    scored = [m for m in per_query if m.get("has_labels")]
    n = len(scored) or 1
    return {
        "recall@k": sum(m["recall@k"] for m in scored) / n,
        "mrr": sum(m["mrr"] for m in scored) / n,
        "hit_rate": sum(m["hit"] for m in scored) / n,
        "n": len(scored),
    }
