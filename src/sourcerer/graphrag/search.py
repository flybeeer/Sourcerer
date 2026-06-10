"""GraphRAG query paths: local search (entity-specific) and global search
(whole-corpus themes, via map-reduce over community summaries).

- **Local search** answers questions about a specific entity: find matching
  entities, gather their descriptions + neighboring relationships, and ground an
  answer in that subgraph.
- **Global search** answers "what are the main themes / overall ..." questions
  that no single chunk contains: map each community summary to a partial answer +
  a helpfulness score, then reduce the helpful ones into a final answer. This is
  what plain vector RAG cannot do.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sourcerer.graphrag.types import GraphIndex, normalize_name
from sourcerer.llm.client import LLMClient

_WORD_RE = re.compile(r"\w+")
_SCORE_RE = re.compile(r"score\s*[:=]\s*(\d{1,3})", re.IGNORECASE)

# A "theme" needs ≥2 related entities; singletons are noise from fragmented
# extraction. Global search maps over the largest such communities only — both to
# stay fast and because whole-corpus answers synthesize the *major* themes.
_MIN_COMMUNITY_SIZE = 2
_MAX_COMMUNITIES = 8

# --- Overview / whole-corpus query detection (router extension) ---
_OVERVIEW_MARKERS = (
    "overall",
    "in general",
    "main theme",
    "main topics",
    "key themes",
    "key topics",
    "across all",
    "whole corpus",
    "high-level",
    "high level",
    "summarize the",
    "summary of all",
    "recurring",
    "what themes",
    "what are the themes",
    "common themes",
    "big picture",
    # Thai
    "ภาพรวม",  # overview
    "ธีมหลัก",  # main themes
    "หัวข้อหลัก",  # main topics
    "โดยรวม",  # overall
    "สรุปทั้งหมด",  # summarize everything
)


def is_overview_query(query: str) -> bool:
    """True if the query asks about whole-corpus themes (→ GraphRAG global)."""
    q = query.lower()
    return any(m in q for m in _OVERVIEW_MARKERS)


@dataclass
class GraphCitation:
    label: str  # e.g. "community 2" or an entity name
    snippet: str
    score: float
    sources: list[str] = field(default_factory=list)  # source files behind it


@dataclass
class GraphResult:
    text: str
    path: str  # "graph-global" | "graph-local"
    citations: list[GraphCitation] = field(default_factory=list)
    input_tokens: int = 0  # totals across all stages (for display)
    output_tokens: int = 0
    # The model that produced the final answer (the reduce step for global), and
    # that stage's tokens — so the caller can cost just the priced portion (map +
    # indexing are local/$0; only the reduce may run on the paid API model).
    model: str = ""
    answer_input_tokens: int = 0
    answer_output_tokens: int = 0


_NO_ANSWER = "I don't know based on the indexed documents."


# --- Local search ---

_LOCAL_PROMPT = """Answer the question using ONLY the entity facts below. If they
do not contain the answer, say exactly: "{no_answer}"

ENTITY FACTS:
{context}

Question: {query}
Answer:"""


def local_search(query: str, index: GraphIndex, client: LLMClient, top_k: int = 5) -> GraphResult:
    """Entity-centric answer: match entities, gather their subgraph, ground on it."""
    terms = {t for t in _WORD_RE.findall(query.lower()) if len(t) > 2}
    scored = []
    for e in index.entities:
        name_terms = set(_WORD_RE.findall(e.name.lower()))
        overlap = len(terms & name_terms)
        if overlap:
            scored.append((overlap, e))
    scored.sort(key=lambda x: x[0], reverse=True)
    hits = [e for _, e in scored[:top_k]]
    if not hits:
        return GraphResult(text=_NO_ANSWER, path="graph-local")

    hit_keys = {normalize_name(e.name) for e in hits}
    rels = [
        r
        for r in index.relationships
        if normalize_name(r.source) in hit_keys or normalize_name(r.target) in hit_keys
    ]
    context = "\n".join(f"- {e.name} ({e.type}): {e.description}" for e in hits)
    if rels:
        context += "\nRelationships:\n" + "\n".join(
            f"- {r.source} → {r.target}: {r.description}" for r in rels[:10]
        )
    result = client.chat(
        [
            {
                "role": "user",
                "content": _LOCAL_PROMPT.format(no_answer=_NO_ANSWER, context=context, query=query),
            }
        ]
    )
    return GraphResult(
        text=result.text,
        path="graph-local",
        citations=[
            GraphCitation(label=e.name, snippet=e.description, score=float(s))
            for s, e in scored[:top_k]
        ],
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        model=result.model,
        answer_input_tokens=result.input_tokens,
        answer_output_tokens=result.output_tokens,
    )


# --- Global search (map-reduce over community summaries) ---

_MAP_PROMPT = """A user asked: "{query}"

Here is a summary of one group of related topics from the documents:
{summary}

How helpful is this group for answering the question? Reply with:
SCORE: <0-100>
POINTS: <the relevant points, or "none">"""

_REDUCE_PROMPT = """Answer the question by synthesizing the points below into a
clear, whole-corpus answer. Use only these points.

Question: {query}

POINTS FROM EACH TOPIC GROUP:
{points}

Answer:"""


def _parse_map(reply: str) -> tuple[int, str]:
    """Extract (score, points) from a map-step reply; tolerant of formatting.

    Small models often skip the SCORE line. Rather than drop the community (which
    can empty the whole answer), default to a neutral score when the reply still
    has substantive points — include-by-default is safer than all-or-nothing.
    """
    points = reply.strip()
    if "POINTS:" in reply.upper():
        idx = reply.upper().index("POINTS:")
        points = reply[idx + len("POINTS:") :].strip()

    m = _SCORE_RE.search(reply)
    if m:
        score = max(0, min(100, int(m.group(1))))
    else:
        # No parseable score: include if there's real content, else drop.
        score = 50 if points and points.lower() not in ("none", "") else 0
    return score, points


def select_communities(index: GraphIndex) -> list:
    """Pick the communities global search maps over: the largest multi-entity
    themes, falling back to all summarized ones for tiny/fragmented graphs."""
    summarized = [c for c in index.communities if c.summary.strip()]
    multi = [c for c in summarized if len(c.entity_names) >= _MIN_COMMUNITY_SIZE]
    chosen = multi or summarized
    return sorted(chosen, key=lambda c: len(c.entity_names), reverse=True)[:_MAX_COMMUNITIES]


def global_search(
    query: str,
    index: GraphIndex,
    client: LLMClient,
    reduce_client: LLMClient | None = None,
) -> GraphResult:
    """Map-reduce over community summaries to answer a whole-corpus question.

    `client` runs the many cheap **map** calls (one per community). `reduce_client`
    (default: `client`) runs the single **reduce** synthesis — pass the frontier
    API client here to spend on the one call that decides answer quality while
    keeping the map bulk local. Phase 4 routing, applied inside GraphRAG.
    """
    reduce_client = reduce_client or client
    communities = select_communities(index)
    if not communities:
        return GraphResult(text=_NO_ANSWER, path="graph-global")

    in_tok = out_tok = 0
    contributions: list[tuple[int, str, object]] = []  # (score, points, community)
    for c in communities:
        res = client.chat(
            [{"role": "user", "content": _MAP_PROMPT.format(query=query, summary=c.summary)}]
        )
        in_tok += res.input_tokens
        out_tok += res.output_tokens
        score, points = _parse_map(res.text)
        if score > 0 and points and points.lower() != "none":
            contributions.append((score, points, c))

    if not contributions:
        return GraphResult(
            text=_NO_ANSWER, path="graph-global", input_tokens=in_tok, output_tokens=out_tok
        )

    contributions.sort(key=lambda x: x[0], reverse=True)
    points_block = "\n".join(f"- (relevance {s}) {p}" for s, p, _ in contributions)
    reduce = reduce_client.chat(
        [{"role": "user", "content": _REDUCE_PROMPT.format(query=query, points=points_block)}]
    )

    return GraphResult(
        text=reduce.text,
        path="graph-global",
        citations=[
            GraphCitation(
                label=f"community {c.id}",
                snippet=c.summary[:240],
                score=float(s),
                sources=c.sources,
            )
            for s, _, c in contributions
        ],
        input_tokens=in_tok + reduce.input_tokens,
        output_tokens=out_tok + reduce.output_tokens,
        model=reduce.model,
        answer_input_tokens=reduce.input_tokens,
        answer_output_tokens=reduce.output_tokens,
    )
