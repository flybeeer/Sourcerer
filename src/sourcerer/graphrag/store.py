"""GraphRAG index storage — a thin facade over two backends.

- **json** (default): one `graph_index.json` under GRAPHRAG_ROOT, loaded whole per
  query. Simple and inspectable; right for small corpora / dev.
- **postgres**: entities/relationships/communities in Postgres with community
  summaries embedded in pgvector, so global search ranks communities by similarity
  instead of size. The production path (see `pgstore`). Selected via GRAPHRAG_STORE.

Callers use the backend-agnostic functions here (`save/load/exists/
select_for_global/indexed_sources`).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from sourcerer.config import Settings
from sourcerer.graphrag import pgstore
from sourcerer.graphrag.types import Community, Entity, GraphIndex, Relationship

_FILENAME = "graph_index.json"


# --- JSON backend ----------------------------------------------------------


def _json_path(root: str | Path) -> Path:
    return Path(root) / _FILENAME


def _json_exists(settings: Settings) -> bool:
    return _json_path(settings.graphrag_root).exists()


def _json_save(index: GraphIndex, settings: Settings) -> None:
    out = _json_path(settings.graphrag_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "entities": [asdict(e) for e in index.entities],
        "relationships": [asdict(r) for r in index.relationships],
        "communities": [asdict(c) for c in index.communities],
    }
    # Atomic write (temp + rename) so a query during a rebuild never reads a
    # half-written file.
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, out)


def _json_load(settings: Settings) -> GraphIndex:
    data = json.loads(_json_path(settings.graphrag_root).read_text(encoding="utf-8"))
    return GraphIndex(
        entities=[Entity(**e) for e in data["entities"]],
        relationships=[Relationship(**r) for r in data["relationships"]],
        communities=[Community(**c) for c in data["communities"]],
    )


# --- Backend-agnostic API --------------------------------------------------


def _is_pg(settings: Settings) -> bool:
    return settings.graphrag_store == "postgres"


def exists(settings: Settings) -> bool:
    return pgstore.exists(settings) if _is_pg(settings) else _json_exists(settings)


def save(index: GraphIndex, settings: Settings) -> None:
    pgstore.save(index, settings) if _is_pg(settings) else _json_save(index, settings)


def load(settings: Settings) -> GraphIndex:
    return pgstore.load(settings) if _is_pg(settings) else _json_load(settings)


def indexed_sources(settings: Settings) -> set[str]:
    """All source files referenced by the stored communities."""
    if _is_pg(settings):
        return pgstore.indexed_sources(settings)
    return {s for c in _json_load(settings).communities for s in c.sources}


def select_for_global(
    query: str, settings: Settings, live_sources: set[str] | None = None, k: int = 8
) -> list[Community]:
    """Pick the communities global search maps over.

    postgres → pgvector similarity to the query (fetches only the top few).
    json → the largest multi-entity communities (loads the whole file first).
    """
    if _is_pg(settings):
        return pgstore.top_communities(query, settings, live_sources, k)
    # Imported here to avoid a circular import at module load.
    from sourcerer.graphrag.search import select_communities

    index = _json_load(settings)
    return select_communities(index, live_sources)
