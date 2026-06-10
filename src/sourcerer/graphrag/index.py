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


def _extraction_client(settings: Settings):
    """The model used for extraction + community summaries.

    Local by default (blueprint: keep extraction local for cost). When
    GRAPHRAG_EXTRACT_WITH_API is on and a key is set, use the frontier API model —
    needed for content a small local model won't structure (e.g. Thai/OCR text it
    summarizes in prose instead of emitting the ENTITY| format).
    """
    if settings.graphrag_extract_with_api and settings.anthropic_api_key:
        from sourcerer.llm.client import get_api_client

        _log.info("GraphRAG extraction using the API model: %s", settings.api_model)
        return get_api_client(settings)
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
    # Traceability: link each community back to its source files.
    from sourcerer import corpus

    graph.attach_community_sources(index, corpus.chunk_sources())
    store.save(index, settings.graphrag_root)
    return index


def rebuild_index(settings: Settings) -> GraphIndex:
    """Refresh the graph after a source deletion, keyed on **source name**.

    Chunk IDs rotate on re-ingestion, so they can't be reconciled against; source
    file names are stable. This drops any community whose every source file was
    deleted and trims deleted files from the rest — permanently removing the
    deleted document's themes (no query-time filtering needed) and promoting the
    next communities into the selectable set. Cheap: no re-extraction, no LLM.
    """
    from sourcerer import corpus

    old = store.load(settings.graphrag_root)
    live = {s["source"] for s in corpus.list_sources()}
    kept = []
    for c in old.communities:
        if c.sources and all(s not in live for s in c.sources):
            continue  # every source deleted → drop the community
        if c.sources:
            c.sources = [s for s in c.sources if s in live]
        kept.append(c)

    index = GraphIndex(entities=old.entities, relationships=old.relationships, communities=kept)
    store.save(index, settings.graphrag_root)
    _log.info("graph rebuilt by source: %d → %d communities", len(old.communities), len(kept))
    return index
