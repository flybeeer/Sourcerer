"""Persist / load the GraphRAG index as JSON artifacts under GRAPHRAG_ROOT.

Plain JSON files (entities/relationships/communities) keep the index inspectable
and avoid extra schema — fine for the small corpus this extension targets.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from sourcerer.graphrag.types import Community, Entity, GraphIndex, Relationship

_FILENAME = "graph_index.json"


def index_path(root: str | Path) -> Path:
    return Path(root) / _FILENAME


def exists(root: str | Path) -> bool:
    return index_path(root).exists()


def save(index: GraphIndex, root: str | Path) -> Path:
    """Write the index to GRAPHRAG_ROOT/graph_index.json."""
    out = index_path(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "entities": [asdict(e) for e in index.entities],
        "relationships": [asdict(r) for r in index.relationships],
        "communities": [asdict(c) for c in index.communities],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load(root: str | Path) -> GraphIndex:
    """Load the index; raises FileNotFoundError if not indexed yet."""
    data = json.loads(index_path(root).read_text(encoding="utf-8"))
    return GraphIndex(
        entities=[Entity(**e) for e in data["entities"]],
        relationships=[Relationship(**r) for r in data["relationships"]],
        communities=[Community(**c) for c in data["communities"]],
    )
