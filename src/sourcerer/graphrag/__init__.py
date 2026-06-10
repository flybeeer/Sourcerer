"""GraphRAG (Phase 6, optional) — a parallel graph-retrieval path.

Whole-corpus / overview questions ("what are the main themes?") that plain vector
RAG can't answer are routed to GraphRAG **global** search (map-reduce over
pre-computed community summaries); entity-specific questions can use **local**
search. Everything is gated behind GRAPHRAG_ENABLED and requires a built index
(see scripts/graphrag_index.py — the expensive step).
"""

from sourcerer.graphrag.search import (
    GraphResult,
    global_search,
    is_overview_query,
    local_search,
)
from sourcerer.graphrag.types import Community, Entity, GraphIndex, Relationship

__all__ = [
    "is_overview_query",
    "global_search",
    "local_search",
    "GraphResult",
    "GraphIndex",
    "Entity",
    "Relationship",
    "Community",
]
