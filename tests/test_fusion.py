"""Tests for reciprocal rank fusion (pure logic, no I/O)."""

from sourcerer.retrieval.fusion import reciprocal_rank_fusion
from sourcerer.retrieval.types import RetrievedChunk


def _chunk(cid: int) -> RetrievedChunk:
    return RetrievedChunk(id=cid, source="s", chunk_index=cid, content=f"c{cid}", score=0.0)


def test_empty_input():
    assert reciprocal_rank_fusion([], k=60) == []
    assert reciprocal_rank_fusion([[], []], k=60) == []


def test_dedupes_by_id():
    a = [_chunk(1), _chunk(2)]
    b = [_chunk(2), _chunk(3)]
    fused = reciprocal_rank_fusion([a, b], k=60)
    assert sorted(c.id for c in fused) == [1, 2, 3]


def test_item_in_both_lists_ranks_first():
    # id=2 appears in both lists, so its summed RRF score should top the result.
    a = [_chunk(1), _chunk(2)]
    b = [_chunk(3), _chunk(2)]
    fused = reciprocal_rank_fusion([a, b], k=60)
    assert fused[0].id == 2


def test_score_is_rrf_sum():
    # id=1 is rank 1 in both lists -> score = 2 * 1/(k+1)
    a = [_chunk(1)]
    b = [_chunk(1)]
    fused = reciprocal_rank_fusion([a, b], k=60)
    assert fused[0].id == 1
    assert abs(fused[0].score - 2 * (1 / 61)) < 1e-9


def test_higher_rank_beats_lower_when_unique():
    a = [_chunk(1), _chunk(2), _chunk(3)]
    fused = reciprocal_rank_fusion([a], k=60)
    assert [c.id for c in fused] == [1, 2, 3]
