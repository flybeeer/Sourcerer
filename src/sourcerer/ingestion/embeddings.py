"""Embedding helpers — a thin layer over the LLM client wrapper.

Keeps the embedding model out of ingestion/retrieval code: callers just ask for
vectors and don't care which backend produced them.
"""

from __future__ import annotations

from sourcerer.llm.client import get_llm_client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts (used during ingestion)."""
    return get_llm_client().embed(texts)


def embed_query(text: str) -> list[float]:
    """Embed a single query string (used at retrieval time)."""
    return get_llm_client().embed([text])[0]
