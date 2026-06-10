"""GraphRAG indexing pipeline — the EXPENSIVE step.

Runs the local model over the whole corpus: one extraction call per chunk, then
one summary call per community. On a small CPU model this is minutes, not
seconds — callers must confirm before running (see scripts/graphrag_index.py).

    chunks → extract entities/relationships (LLM × n_chunks)
           → merge + detect communities
           → summarize each community (LLM × n_communities)
           → persist GraphIndex to GRAPHRAG_ROOT
"""

from __future__ import annotations

import logging

from sourcerer.config import Settings
from sourcerer.graphrag import graph, store, summarize
from sourcerer.graphrag.extraction import extract_from_chunk
from sourcerer.graphrag.types import GraphIndex, normalize_name
from sourcerer.llm.ollama_client import OllamaClient

_log = logging.getLogger(__name__)


def _extraction_client(settings: Settings) -> OllamaClient:
    """The LOCAL model used for extraction + summaries (never the frontier API)."""
    return OllamaClient(
        base_url=settings.ollama_base_url,
        chat_model=settings.graphrag_extraction_model,
        embedding_model=settings.embedding_model,
    )


def estimate(chunks: list[tuple[int, str]], n_communities_guess: int = 5) -> dict:
    """Rough cost estimate (LLM calls) without running anything."""
    return {
        "chunks": len(chunks),
        "extraction_calls": len(chunks),
        "summary_calls_est": n_communities_guess,
        "total_llm_calls_est": len(chunks) + n_communities_guess,
    }


def build_index(chunks: list[tuple[int, str]], settings: Settings) -> GraphIndex:
    """Run the full pipeline over (chunk_id, text) pairs. Expensive — see module doc."""
    client = _extraction_client(settings)

    all_entities = []
    all_relationships = []
    for i, (chunk_id, text) in enumerate(chunks, start=1):
        _log.info("extracting %d/%d (chunk %d)", i, len(chunks), chunk_id)
        ents, rels = extract_from_chunk(text, chunk_id, client)
        all_entities.extend(ents)
        all_relationships.extend(rels)

    entities = graph.merge_entities(all_entities)
    communities = graph.detect_communities(entities, all_relationships)
    _log.info(
        "graph: %d entities, %d relationships, %d communities",
        len(entities),
        len(all_relationships),
        len(communities),
    )

    # Only summarize real themes (≥2 related entities). Singletons from fragmented
    # extraction add cost and noise; leaving their summary empty excludes them from
    # global search. Fall back to summarizing all if nothing is multi-entity.
    by_name = {normalize_name(e.name): e for e in entities}
    to_summarize = [c for c in communities if len(c.entity_names) >= 2] or communities
    for c in to_summarize:
        _log.info("summarizing community %d (%d entities)", c.id, len(c.entity_names))
        c.summary = summarize.summarize_community(c, by_name, all_relationships, client)

    index = GraphIndex(entities=entities, relationships=all_relationships, communities=communities)
    store.save(index, settings.graphrag_root)
    return index
