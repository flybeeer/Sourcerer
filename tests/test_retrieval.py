"""Tests for retrieval metrics (pure logic, no I/O)."""

from sourcerer.eval.retrieval_metrics import (
    aggregate,
    hit,
    per_query_metrics,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k():
    retrieved = ["a.md", "b.md", "c.md"]
    assert recall_at_k(retrieved, {"a.md"}, k=3) == 1.0
    assert recall_at_k(retrieved, {"a.md", "x.md"}, k=3) == 0.5
    assert recall_at_k(retrieved, {"a.md"}, k=0) == 0.0
    assert recall_at_k(retrieved, set(), k=3) == 0.0  # no labels -> 0


def test_reciprocal_rank():
    retrieved = ["x.md", "a.md", "b.md"]
    assert reciprocal_rank(retrieved, {"a.md"}, k=3) == 0.5  # first relevant at rank 2
    assert reciprocal_rank(["a.md"], {"a.md"}, k=3) == 1.0
    assert reciprocal_rank(retrieved, {"none.md"}, k=3) == 0.0


def test_hit():
    retrieved = ["a.md", "b.md"]
    assert hit(retrieved, {"b.md"}, k=2) == 1.0
    assert hit(retrieved, {"b.md"}, k=1) == 0.0  # b.md is rank 2, outside k=1
    assert hit(retrieved, {"z.md"}, k=2) == 0.0


def test_dedupes_sources_in_recall():
    # duplicate sources in the retrieved list shouldn't inflate recall
    retrieved = ["a.md", "a.md", "a.md"]
    assert recall_at_k(retrieved, {"a.md", "b.md"}, k=3) == 0.5


def test_aggregate_ignores_unlabeled():
    per_query = [
        per_query_metrics(["a.md"], ["a.md"], k=3),  # perfect
        per_query_metrics(["x.md"], ["a.md"], k=3),  # miss
        per_query_metrics(["a.md"], [], k=3),  # no labels -> ignored
    ]
    agg = aggregate(per_query)
    assert agg["n"] == 2
    assert agg["recall@k"] == 0.5
    assert agg["hit_rate"] == 0.5
    assert agg["mrr"] == 0.5
