"""GraphRAG data types — the knowledge graph + community summaries.

A deliberately small, explicit model: entities (nodes), relationships (edges),
and communities (clusters of entities with an LLM-written summary). These are the
artifacts the indexing pipeline produces and the search paths consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def normalize_name(name: str) -> str:
    """Canonical key for an entity name (dedupe across chunks)."""
    return " ".join(name.strip().lower().split())


@dataclass
class Entity:
    name: str
    type: str
    description: str
    # Source chunks this entity was extracted from (for local-search context).
    chunk_ids: list[int] = field(default_factory=list)


@dataclass
class Relationship:
    source: str  # entity name
    target: str  # entity name
    description: str


@dataclass
class Community:
    id: int
    entity_names: list[str]
    summary: str = ""
    # Source files this community's entities were extracted from (traceability).
    sources: list[str] = field(default_factory=list)
    # Summary embedding — set by the json store so global search can rank by
    # similarity (postgres keeps embeddings in pgvector instead). Empty = unranked.
    embedding: list[float] = field(default_factory=list)


@dataclass
class GraphIndex:
    """The full set of artifacts produced by indexing."""

    entities: list[Entity]
    relationships: list[Relationship]
    communities: list[Community]

    def entity_by_name(self) -> dict[str, Entity]:
        return {normalize_name(e.name): e for e in self.entities}
