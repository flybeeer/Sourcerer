"""LLM-written community reports (the pre-computed "global" summaries).

For each community, the local model writes a short report from the community's
entities and relationships. These reports are what global search map-reduces over
at query time — the whole point of GraphRAG: the corpus-level structure is
summarized *ahead* of any question.
"""

from __future__ import annotations

from sourcerer.graphrag.types import Community, Entity, Relationship, normalize_name
from sourcerer.llm.client import LLMClient

_PROMPT = """Write a concise report (3-5 sentences) describing this group of
related entities from a company's documents. State the main theme and the key
facts. Use only the information given.

ENTITIES:
{entities}

RELATIONSHIPS:
{relationships}

REPORT:"""


def build_prompt(entities: list[Entity], relationships: list[Relationship]) -> str:
    ents = "\n".join(f"- {e.name} ({e.type}): {e.description}" for e in entities) or "(none)"
    rels = (
        "\n".join(f"- {r.source} → {r.target}: {r.description}" for r in relationships) or "(none)"
    )
    return _PROMPT.format(entities=ents, relationships=rels)


def summarize_community(
    community: Community,
    entities_by_name: dict[str, Entity],
    relationships: list[Relationship],
    client: LLMClient,
) -> str:
    """One LLM call: write the report for a single community."""
    members = {normalize_name(n) for n in community.entity_names}
    ents = [entities_by_name[n] for n in members if n in entities_by_name]
    rels = [
        r
        for r in relationships
        if normalize_name(r.source) in members and normalize_name(r.target) in members
    ]
    return client.chat([{"role": "user", "content": build_prompt(ents, rels)}]).text.strip()
