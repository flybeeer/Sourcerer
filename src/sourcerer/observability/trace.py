"""Structured per-query logging (Phase 5 observability).

Emits one JSON object per query on the `sourcerer.query` logger, carrying the
full trace: the query, retrieval mode + which chunks were retrieved (source,
chunk index, score), the route taken and why, token usage, latency, and cost.
This is the machine-readable companion to the `query_log` table — pipe stdout to
a log collector and every field is queryable.
"""

from __future__ import annotations

import json
import logging

from sourcerer.retrieval.types import RetrievedChunk

_logger = logging.getLogger("sourcerer.query")


def retrieval_trace(chunks: list[RetrievedChunk]) -> list[dict]:
    """Compact, JSON-safe view of what retrieval returned."""
    return [
        {"source": c.source, "chunk_index": c.chunk_index, "score": round(c.score, 4)}
        for c in chunks
    ]


def log_query_event(
    *,
    query: str,
    retrieval_mode: str,
    chunks: list[RetrievedChunk],
    route: str,
    model: str,
    router_reason: str,
    difficulty: float,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    latency_ms: int,
    answered: bool,
    guardrail: str | None = None,
    principal: str | None = None,
    denied_assets: int | None = None,
) -> None:
    """Emit one structured JSON log line for an answered (or refused) query.

    `principal` (Phase 10a) is the identity that issued the query; `denied_assets`
    (Phase 10d) is how many assets the gate hid. Both None when governance is off.
    """
    event = {
        "event": "query",
        "query": query,
        "principal": principal,
        "denied_assets": denied_assets,
        "retrieval_mode": retrieval_mode,
        "retrieved": retrieval_trace(chunks),
        "num_retrieved": len(chunks),
        "route": route,
        "model": model,
        "router_reason": router_reason,
        "difficulty": difficulty,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost_usd, 6),
        "latency_ms": latency_ms,
        "answered": answered,
        "guardrail": guardrail,
    }
    _logger.info(json.dumps(event, ensure_ascii=False))
