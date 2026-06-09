"""The single LLM client interface.

Per the project's hard rule, *all* LLM calls go through one thin wrapper so the
backend (local Ollama now; vLLM / frontier API later) is swappable without
touching ingestion, retrieval, or generation code.

Phase 1 has exactly one backend: Ollama. `get_llm_client()` is the only entry
point callers should use.
"""

from __future__ import annotations

from typing import Protocol

from sourcerer.config import get_settings


class Message(dict):
    """A chat message: {"role": "system"|"user"|"assistant", "content": str}."""


class LLMClient(Protocol):
    """Minimal surface every backend must implement."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...

    def chat(self, messages: list[dict]) -> str:
        """Return the assistant's reply text for a list of chat messages."""
        ...


def get_llm_client() -> LLMClient:
    """Return the configured LLM client. Phase 1: always the local Ollama client."""
    # Imported lazily to avoid a circular import (ollama_client imports config).
    from sourcerer.llm.ollama_client import OllamaClient

    settings = get_settings()
    return OllamaClient(
        base_url=settings.ollama_base_url,
        chat_model=settings.local_model,
        embedding_model=settings.embedding_model,
    )
