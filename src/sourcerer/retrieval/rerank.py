"""Reranking stage: re-score fused candidates and keep the best top_k.

Three backends, selected by RERANKER_TYPE:
- "none"   : passthrough — keep the fused order, just truncate to top_k.
- "local"  : a local cross-encoder (bge-reranker) via sentence-transformers.
- "cohere" : Cohere's Rerank API.

The "local" and "cohere" backends import their (heavy / optional) dependencies
lazily, so the package installs and runs with just the "none" backend. Install
extras as needed:  pip install -e ".[rerank]"   or   pip install -e ".[cohere]"
"""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from sourcerer.config import Settings
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

    def _ensure_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise RuntimeError(
                    "Local reranker needs sentence-transformers. "
                    'Install it with: pip install -e ".[rerank]"'
                ) from exc
            self._model = CrossEncoder(self.model_name)
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


def get_reranker(settings: Settings) -> Reranker:
    """Build the reranker selected by RERANKER_TYPE."""
    kind = settings.reranker_type.lower()
    if kind == "none":
        return NoopReranker()
    if kind == "local":
        return LocalReranker(settings.reranker_model)
    if kind == "cohere":
        return CohereReranker(settings.reranker_model, settings.cohere_api_key)
    raise ValueError(f"Unknown RERANKER_TYPE: {settings.reranker_type!r}")
