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


def test_attach_community_sources():
    ents = [
        Entity("Alpha", "concept", "d", chunk_ids=[1, 2]),
        Entity("Beta", "concept", "d", chunk_ids=[3]),
    ]
    idx = GraphIndex(
        entities=ents,
        relationships=[],
        communities=[Community(0, ["Alpha", "Beta"]), Community(1, ["Alpha"])],
    )
    graph.attach_community_sources(idx, {1: "a.md", 2: "b.md", 3: "c.md"})
    assert idx.communities[0].sources == ["a.md", "b.md", "c.md"]
    assert idx.communities[1].sources == ["a.md", "b.md"]


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
    # The reduce stage's usage is tracked separately for costing.
    assert res.answer_input_tokens == 10 and res.answer_output_tokens == 5
    assert res.model == "fake"


def test_global_search_reduce_uses_separate_client():
    # Map runs on the local client; reduce runs on a distinct (API) client.
    map_client = FakeClient(["SCORE: 80\nPOINTS: a point", "SCORE: 60\nPOINTS: another"])

    class ApiClient(FakeClient):
        pass

    reduce_client = ApiClient(["synthesized answer"])
    reduce_client._replies = ["synthesized answer"]
    # Mark the reduce client's model so we can assert it produced the answer.
    reduce_client.chat = lambda messages: ChatResult(
        text="synthesized answer", model="claude-sonnet-4-6", input_tokens=99, output_tokens=7
    )

    res = search.global_search(
        "main themes?", _index_with_two_communities(), map_client, reduce_client=reduce_client
    )
    assert map_client.calls == 2  # only the map calls hit the local client
    assert res.text == "synthesized answer"
    assert res.model == "claude-sonnet-4-6"
    assert res.answer_input_tokens == 99 and res.answer_output_tokens == 7


def test_select_communities_drops_deleted_sources():
    idx = GraphIndex(
        entities=[],
        relationships=[],
        communities=[
            Community(0, ["A", "B"], summary="x", sources=["keep.md"]),
            Community(1, ["C", "D"], summary="y", sources=["deleted.md"]),
            Community(2, ["E", "F"], summary="z", sources=["keep.md", "deleted.md"]),
        ],
    )
    # Only keep.md survives → community 1 (deleted-only) is dropped; 0 and 2 stay.
    kept = search.select_communities(idx, live_sources={"keep.md"})
    assert {c.id for c in kept} == {0, 2}


def test_global_search_filters_deleted_source_from_citations():
    idx = GraphIndex(
        entities=[],
        relationships=[],
        communities=[Community(0, ["A", "B"], summary="theme", sources=["keep.md", "gone.md"])],
    )
    client = FakeClient(["SCORE: 90\nPOINTS: a point", "final answer"])
    res = search.global_search("themes?", idx, client, live_sources={"keep.md"})
    assert res.citations[0].sources == ["keep.md"]  # gone.md filtered out


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
