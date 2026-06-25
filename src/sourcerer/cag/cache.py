"""Build + memoize governance-partitioned context caches (CAG PoC).

A "cache" here is the assembled context string (the documents a principal may
read, concatenated) — the thing CAG preloads instead of retrieving. Because the
context prefix is stable per access profile, a KV-caching backend (llama.cpp /
vLLM prompt cache) reuses the computed attention across queries: that's the CAG
speedup. Cerbos decides cache membership before any content is assembled.

Documents are read **straight from disk** (CAG_CORPUS_DIR), not the `chunks`
table: CAG wants whole documents and never embeds anything, so it needs neither
chunking nor the ingestion pipeline — only the full text. The file's basename is
its source/asset id (matching how `ingest_directory` labels sources), so the same
catalog tags and Cerbos policy govern it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from sourcerer import governance
from sourcerer.config import Settings
from sourcerer.governance import plan
from sourcerer.governance.principal import Principal
from sourcerer.ingestion.loaders import SUPPORTED_SUFFIXES, load_document


def _disk_index(settings: Settings) -> dict[str, Path]:
    """Map source id (file basename) → path for every supported file on disk.

    Basename keys mirror `ingestion.pipeline.ingest_directory`, which labels each
    source by `path.name` — so a source authorized by the catalog/Cerbos resolves
    to the right file regardless of how the corpus dir is nested.
    """
    root = Path(settings.cag_corpus_dir)
    if not root.is_dir():
        return {}
    return {
        path.name: path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    }


def load_full_documents(sources: set[str], settings: Settings) -> dict[str, str]:
    """Read each source's full text from disk (skips missing/empty files)."""
    if not sources:
        return {}
    index = _disk_index(settings)
    docs: dict[str, str] = {}
    for source in sorted(sources):
        path = index.get(source)
        if path is None:
            continue  # authorized but not on disk → simply unavailable, never fatal
        text = load_document(path)
        if text:
            docs[source] = text
    return docs


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


def _gate_sources(
    principal: Principal | None, settings: Settings, universe: set[str]
) -> set[str] | None:
    """Readable source set for `principal`, honouring the pushdown flag.

    GOVERNANCE_PUSHDOWN (with the Cerbos PDP) asks Cerbos PlanResources once and
    translates the plan to a SQL WHERE over the indexed corpus; otherwise the
    CheckResources/local path authorizes the on-disk `universe` in-process. Both
    return the same readable set (asserted by scripts/plan_parity.py), but the
    plan path authorizes over the `chunks` corpus, which can differ from disk — so
    the result is intersected with `universe` to keep CAG to what it can serve.
    Returns None when governance is off (no filter), matching either path.
    """
    if settings.governance_pushdown and settings.governance_pdp == "cerbos":
        allowed = plan.allowed_document_sources_plan(principal, settings)
    else:
        allowed = governance.allowed_document_sources(principal, settings, sources=universe)
    return allowed if allowed is None else allowed & universe


def build_cache(
    principal: Principal | None,
    settings: Settings,
    *,
    limit_sources: set[str] | None = None,
) -> ContextCache:
    """Build (or reuse) the context cache for `principal`'s access profile.

    The Cerbos gate runs here, at build time, over the on-disk corpus: `_gate_sources`
    returns the sources this principal may read (None = governance off → everything
    on disk), via CheckResources or — under GOVERNANCE_PUSHDOWN — PlanResources
    predicate pushdown. Forbidden sources are never loaded, so they can't reach the
    model's context. `limit_sources` optionally narrows the universe (e.g. to a
    small demo corpus that fits in context).
    """
    universe = set(_disk_index(settings))
    allowed = _gate_sources(principal, settings, universe)
    governed = allowed is not None
    if allowed is None:  # governance off → the whole (optionally limited) disk corpus
        allowed = limit_sources if limit_sources is not None else universe
    elif limit_sources is not None:
        allowed = allowed & limit_sources

    key = frozenset(allowed)
    if key in _STORE:
        _STORE[key].hits += 1
        return _STORE[key]

    started = time.perf_counter()
    docs = load_full_documents(allowed, settings)
    cache = ContextCache(
        sources=sorted(docs),
        context=_assemble(docs),
        build_ms=int((time.perf_counter() - started) * 1000),
        governed=governed,
    )
    _STORE[key] = cache
    return cache


def clear() -> None:
    """Drop all cached contexts (PoC helper)."""
    _STORE.clear()
