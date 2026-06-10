"""Entity + relationship extraction from a chunk, using the LOCAL model.

GraphRAG's indexing step is the expensive one — it runs an LLM over every chunk.
Per the project's cost rule we use the *local* model (GRAPHRAG_EXTRACTION_MODEL),
not the frontier API, so whole-corpus indexing stays free.

Output format is line-based and pipe-delimited (not JSON) — small local models
emit it far more reliably than nested JSON, and `parse_extraction()` is pure and
unit-testable without any model call.
"""

from __future__ import annotations

from sourcerer.graphrag.types import Entity, Relationship
from sourcerer.llm.client import LLMClient

_PROMPT = """You are building a knowledge graph from a document.

From the text below, extract:
- ENTITIES: important people, teams, systems, policies, places, or concepts.
- RELATIONSHIPS: how two entities are connected.

Output ONLY lines in these exact formats, nothing else:
ENTITY|<name>|<type>|<one-sentence description>
RELATIONSHIP|<source entity>|<target entity>|<how they relate>

Use a short TYPE like person, team, system, policy, place, concept.
Do not add commentary, headers, or blank explanations.

TEXT:
{text}
"""


def build_prompt(text: str) -> str:
    return _PROMPT.format(text=text)


def parse_extraction(reply: str, chunk_id: int) -> tuple[list[Entity], list[Relationship]]:
    """Parse the model's line-based reply into entities and relationships.

    Tolerant: ignores malformed lines and surrounding prose, trims fields, and
    drops rows missing required parts.
    """
    entities: list[Entity] = []
    relationships: list[Relationship] = []
    for raw in reply.splitlines():
        line = raw.strip().lstrip("-* ").strip()
        if not line or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        tag = parts[0].upper()
        if tag == "ENTITY" and len(parts) >= 4 and parts[1]:
            entities.append(
                Entity(
                    name=parts[1],
                    type=parts[2] or "concept",
                    description=parts[3],
                    chunk_ids=[chunk_id],
                )
            )
        elif tag == "RELATIONSHIP" and len(parts) >= 4 and parts[1] and parts[2]:
            relationships.append(
                Relationship(source=parts[1], target=parts[2], description=parts[3])
            )
    return entities, relationships


def extract_from_chunk(
    text: str, chunk_id: int, client: LLMClient
) -> tuple[list[Entity], list[Relationship]]:
    """One LLM call: extract entities + relationships from a chunk."""
    reply = client.chat([{"role": "user", "content": build_prompt(text)}]).text
    return parse_extraction(reply, chunk_id)
