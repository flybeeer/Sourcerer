"""Tests for GraphRAG (Phase 6) — pure logic + orchestration with a fake client.

No real LLM calls and no indexing run here; the expensive pipeline is exercised
only through a canned-reply fake client.
"""

from sourcerer.graphrag import graph, search
from sourcerer.graphrag.extraction import parse_extraction
from sourcerer.graphrag.types import Community, Entity, GraphIndex, Relationship
from sourcerer.llm.client import ChatResult


class FakeClient:
    """Returns canned replies in order (or a single reply repeatedly)."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0

    def embed(self, texts):  # pragma: no cover - unused
        raise NotImplementedError

    def chat(self, messages):
        reply = self._replies[min(self.calls, len(self._replies) - 1)]
        self.calls += 1
        return ChatResult(text=reply, model="fake", input_tokens=10, output_tokens=5)


# ---- extraction parsing ----


def test_parse_extraction_well_formed():
    reply = (
        "ENTITY|On-Call Rotation|team|Engineers who handle incidents\n"
        "ENTITY|Stipend|policy|Weekly pay for on-call\n"
        "RELATIONSHIP|On-Call Rotation|Stipend|receives a weekly stipend"
    )
    ents, rels = parse_extraction(reply, chunk_id=3)
    assert [e.name for e in ents] == ["On-Call Rotation", "Stipend"]
    assert ents[0].chunk_ids == [3]
    assert rels[0].source == "On-Call Rotation" and rels[0].target == "Stipend"


def test_parse_extraction_ignores_garbage():
    reply = "Here is what I found:\nENTITY|Alpha|system|does things\nnot a real line\nENTITY|bad"
    ents, rels = parse_extraction(reply, chunk_id=1)
    assert [e.name for e in ents] == ["Alpha"]
    assert rels == []


# ---- graph build + communities ----


def test_merge_entities_dedupes_and_unions():
    a = Entity("Sourcerer", "company", "short", chunk_ids=[1])
    b = Entity("sourcerer", "company", "a longer description wins", chunk_ids=[2])
    merged = graph.merge_entities([a, b])
    assert len(merged) == 1
    assert merged[0].chunk_ids == [1, 2]
    assert merged[0].description == "a longer description wins"


def test_detect_communities_splits_disconnected_groups():
    ents = [Entity(n, "concept", "d") for n in ["A", "B", "C", "D"]]
    rels = [Relationship("A", "B", "r"), Relationship("C", "D", "r")]
    comms = graph.detect_communities(ents, rels)
    # Two disconnected pairs → at least two communities.
    assert len(comms) >= 2
    members = {frozenset(c.entity_names) for c in comms}
    assert frozenset({"A", "B"}) in members or frozenset({"C", "D"}) in members


def test_detect_communities_empty():
    assert graph.detect_communities([], []) == []


# ---- overview detection (router extension) ----


def test_overview_detection():
    assert search.is_overview_query("What are the main themes across all documents?")
    assert search.is_overview_query("Give a high-level overview of the policies.")
    assert search.is_overview_query("ภาพรวมของเอกสารทั้งหมดคืออะไร")
    assert not search.is_overview_query("What is the on-call stipend?")
    assert not search.is_overview_query("Where is the headquarters?")


# ---- global search map-reduce orchestration (fake client) ----


def _index_with_two_communities():
    return GraphIndex(
        entities=[],
        relationships=[],
        communities=[
            Community(0, ["A"], summary="HR policies: PTO and benefits."),
            Community(1, ["B"], summary="Engineering on-call and incidents."),
        ],
    )


def test_global_search_map_reduce():
    # 2 map calls (both helpful) + 1 reduce call.
    client = FakeClient(
        [
            "SCORE: 80\nPOINTS: HR covers PTO and benefits.",
            "SCORE: 70\nPOINTS: Engineering covers on-call.",
            "The documents cover HR and engineering operations.",
        ]
    )
    res = search.global_search("main themes?", _index_with_two_communities(), client)
    assert client.calls == 3  # map, map, reduce
    assert res.path == "graph-global"
    assert "HR and engineering" in res.text
    assert len(res.citations) == 2
    assert res.input_tokens == 30 and res.output_tokens == 15


def test_global_search_no_helpful_communities():
    client = FakeClient(["SCORE: 0\nPOINTS: none"])
    res = search.global_search("unrelated?", _index_with_two_communities(), client)
    assert res.citations == []
    assert "don't know" in res.text.lower()


def test_parse_map_extracts_score_and_points():
    score, points = search._parse_map("SCORE: 42\nPOINTS: the key fact")
    assert score == 42
    assert points == "the key fact"


def test_parse_map_defaults_to_include_when_no_score():
    # Small models often skip SCORE; substantive content should still be included.
    score, points = search._parse_map("This group covers benefits and PTO.")
    assert score == 50
    assert "benefits" in points


def test_parse_map_drops_empty_or_none():
    assert search._parse_map("POINTS: none")[0] == 0
    assert search._parse_map("")[0] == 0
