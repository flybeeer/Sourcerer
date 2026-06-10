"""Reranking stage: re-score fused candidates and keep the best top_k.

Four backends, selected by RERANKER_TYPE:
- "none"   : passthrough — keep the fused order, just truncate to top_k.
- "local"  : a local cross-encoder (bge-reranker) via sentence-transformers.
- "cohere" : Cohere's Rerank API.
- "llm"    : listwise reranking by the local LLM (one call, no extra deps).

The "local" and "cohere" backends import their (heavy / optional) dependencies
lazily, so the package installs and runs with just the "none" / "llm" backends.
Install extras as needed:  pip install -e ".[rerank]"  or  pip install -e ".[cohere]"
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Protocol

from sourcerer.config import Settings
from sourcerer.llm.ollama_client import OllamaClient
from sourcerer.retrieval.types import RetrievedChunk


class Reranker(Protocol):
    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]: ...


class NoopReranker:
    """Keep the incoming (fused) order; just truncate."""

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        return chunks[:top_k]


class LocalReranker:
    """Local cross-encoder reranker (bge-reranker via sentence-transformers)."""

    def __init__(self, model: str) -> None:
        # Default config stores a short name; map it to the HF repo id.
        self.model_name = model if "/" in model else f"BAAI/{model}"
        self._model = None  # loaded lazily on first use

    @staticmethod
    def _best_device() -> str:
        """Pick the inference device — CUDA if present, otherwise CPU.

        We deliberately force CPU over Apple MPS: on this torch build a large
        cross-encoder deadlocks inside Metal (`MPSStream::synchronize` never
        returns), and sentence-transformers auto-selects MPS unless told not to.
        CPU is fast enough for a *small* cross-encoder — pair this with a small
        model (e.g. cross-encoder/ms-marco-MiniLM-L6-v2), not a 500M+ one.
        """
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _ensure_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise RuntimeError(
                    "Local reranker needs sentence-transformers. "
                    'Install it with: pip install -e ".[rerank]"'
                ) from exc
            self._model = CrossEncoder(self.model_name, device=self._best_device())
        return self._model

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        model = self._ensure_model()
        scores = model.predict([(query, c.content) for c in chunks])
        rescored = [replace(c, score=float(s)) for c, s in zip(chunks, scores, strict=True)]
        rescored.sort(key=lambda c: c.score, reverse=True)
        return rescored[:top_k]


class CohereReranker:
    """Cohere Rerank API."""

    def __init__(self, model: str, api_key: str) -> None:
        if not api_key:
            raise RuntimeError("RERANKER_TYPE=cohere requires COHERE_API_KEY in .env")
        self.model = model
        self.api_key = api_key
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            try:
                import cohere
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise RuntimeError(
                    'Cohere reranker needs the cohere package. Install: pip install -e ".[cohere]"'
                ) from exc
            self._client = cohere.Client(self.api_key)
        return self._client

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        client = self._ensure_client()
        result = client.rerank(
            model=self.model,
            query=query,
            documents=[c.content for c in chunks],
            top_n=min(top_k, len(chunks)),
        )
        return [replace(chunks[r.index], score=float(r.relevance_score)) for r in result.results]


class LLMReranker:
    """Listwise reranking by the local LLM — one call, no extra dependencies.

    The model is shown the numbered candidates and asked to return the most
    relevant candidate numbers in order. Falls back to the fused order if the
    reply can't be parsed.
    """

    # Cap candidates sent to the LLM to keep the prompt and latency bounded.
    _MAX_CANDIDATES = 12
    _SNIPPET_CHARS = 300

    def __init__(self, base_url: str, model: str) -> None:
        self._client = OllamaClient(base_url=base_url, chat_model=model, embedding_model="")

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        candidates = chunks[: self._MAX_CANDIDATES]
        listing = "\n".join(
            f"[{i}] {c.content[: self._SNIPPET_CHARS]}" for i, c in enumerate(candidates, start=1)
        )
        prompt = (
            "Rank the candidate passages by how well they help answer the question.\n\n"
            f"Question: {query}\n\nCandidates:\n{listing}\n\n"
            f"Return the numbers of the {top_k} most relevant candidates, best first, "
            "as a comma-separated list (e.g. 3,1,5). Numbers only."
        )
        reply = self._client.chat([{"role": "user", "content": prompt}]).text

        order = [int(n) for n in re.findall(r"\d+", reply)]
        seen: set[int] = set()
        ranked: list[RetrievedChunk] = []
        for n in order:
            if 1 <= n <= len(candidates) and n not in seen:
                seen.add(n)
                ranked.append(candidates[n - 1])
        # Append any candidates the model omitted, preserving fused order.
        for i, c in enumerate(candidates, start=1):
            if i not in seen:
                ranked.append(c)
        return ranked[:top_k]


# Cache reranker instances so heavy models (the local cross-encoder) load once
# per process, not once per request. Keyed by the settings that define a backend.
_RERANKER_CACHE: dict[tuple, Reranker] = {}


def _build_reranker(settings: Settings) -> Reranker:
    kind = settings.reranker_type.lower()
    if kind == "none":
        return NoopReranker()
    if kind == "local":
        return LocalReranker(settings.reranker_model)
    if kind == "cohere":
        return CohereReranker(settings.reranker_model, settings.cohere_api_key)
    if kind == "llm":
        return LLMReranker(settings.ollama_base_url, settings.local_model)
    raise ValueError(f"Unknown RERANKER_TYPE: {settings.reranker_type!r}")


def get_reranker(settings: Settings) -> Reranker:
    """Return the configured reranker, cached so its model loads only once."""
    key = (
        settings.reranker_type.lower(),
        settings.reranker_model,
        settings.ollama_base_url,
        settings.local_model,
    )
    if key not in _RERANKER_CACHE:
        _RERANKER_CACHE[key] = _build_reranker(settings)
    return _RERANKER_CACHE[key]
