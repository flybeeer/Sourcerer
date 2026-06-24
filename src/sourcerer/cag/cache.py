"""Build + memoize governance-partitioned context caches (CAG PoC).

A "cache" here is the assembled context string (the documents a principal may
read, concatenated) — the thing CAG preloads instead of retrieving. Because the
context prefix is stable per access profile, a KV-caching backend (llama.cpp /
vLLM prompt cache) reuses the computed attention across queries: that's the CAG
speedup. Cerbos decides cache membership before any content is assembled.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from sourcerer import governance
from sourcerer.config import Settings
from sourcerer.db.session import connect
from sourcerer.governance.principal import Principal


def load_full_documents(sources: set[str]) -> dict[str, str]:
    """Reassemble each source's full text from its chunks (ordered)."""
    if not sources:
        return {}
    with connect() as conn:
        rows = conn.execute(
            "SELECT source, content FROM chunks "
            "WHERE source = ANY(%s) ORDER BY source, chunk_index",
            (list(sources),),
        ).fetchall()
    docs: dict[str, list[str]] = {}
    for source, content in rows:
        docs.setdefault(source, []).append(content)
    return {s: "\n".join(parts) for s, parts in docs.items()}


@dataclass
class ContextCache:
    """A preloaded, access-scoped context (the CAG "cache")."""

    sources: list[str]  # the sources baked in (sorted) — the access profile key
    context: str  # the assembled prompt context
    build_ms: int = 0
    hits: int = 0  # how many answers reused this cache (cache-reuse signal)
    governed: bool = True  # False when governance was off (full corpus)


# In-memory store keyed by the *allowed-source set*, so two principals with the
# same access share one cache (per-tier, not per-user).
_STORE: dict[frozenset[str], ContextCache] = {}


def _assemble(docs: dict[str, str]) -> str:
    """Concatenate documents into one context block with source headers."""
    return "\n\n".join(f"[source: {s}]\n{text}" for s, text in sorted(docs.items()))


def build_cache(
    principal: Principal | None,
    settings: Settings,
    *,
    limit_sources: set[str] | None = None,
) -> ContextCache:
    """Build (or reuse) the context cache for `principal`'s access profile.

    The Cerbos gate runs here, at build time: `allowed_document_sources` returns the
    sources this principal may read (None = governance off → everything). Forbidden
    sources are never loaded, so they can't reach the model's context. `limit_sources`
    optionally narrows the universe (e.g. to a small demo corpus that fits in context).
    """
    allowed = governance.allowed_document_sources(principal, settings)
    governed = allowed is not None
    if allowed is None:  # governance off → the whole (optionally limited) corpus
        allowed = limit_sources or _all_sources()
    elif limit_sources is not None:
        allowed = allowed & limit_sources

    key = frozenset(allowed)
    if key in _STORE:
        _STORE[key].hits += 1
        return _STORE[key]

    started = time.perf_counter()
    docs = load_full_documents(allowed)
    cache = ContextCache(
        sources=sorted(docs),
        context=_assemble(docs),
        build_ms=int((time.perf_counter() - started) * 1000),
        governed=governed,
    )
    _STORE[key] = cache
    return cache


def _all_sources() -> set[str]:
    with connect() as conn:
        return {r[0] for r in conn.execute("SELECT DISTINCT source FROM chunks").fetchall()}


def clear() -> None:
    """Drop all cached contexts (PoC helper)."""
    _STORE.clear()
