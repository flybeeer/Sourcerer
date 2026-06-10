"""The single LLM client interface.

Per the project's hard rule, *all* LLM calls go through one thin wrapper so the
backend (local Ollama, or the frontier API for the Phase 4 "hard query" route)
is swappable without touching ingestion, retrieval, or generation code.

- `get_llm_client()` — the local Ollama client. Embeddings always run here, and
  it is the default generation backend (easy/sensitive/high-volume queries).
- `get_api_client()` — the frontier API client for the hard-query route.
- `client_for(route)` — pick the backend for a router decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from sourcerer.config import Settings, get_settings


@dataclass
class ChatResult:
    """A chat completion plus the usage needed for cost/latency instrumentation."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int


class Message(dict):
    """A chat message: {"role": "system"|"user"|"assistant", "content": str}."""


class LLMClient(Protocol):
    """Minimal surface every backend must implement."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...

    def chat(self, messages: list[dict]) -> ChatResult:
        """Return the assistant's reply (with token usage) for a list of messages."""
        ...


def get_llm_client(settings: Settings | None = None) -> LLMClient:
    """Return the local Ollama client (embeddings + the local generation route)."""
    # Imported lazily to avoid a circular import (ollama_client imports config).
    from sourcerer.llm.ollama_client import OllamaClient

    settings = settings or get_settings()
    return OllamaClient(
        base_url=settings.ollama_base_url,
        chat_model=settings.local_model,
        embedding_model=settings.embedding_model,
    )


def get_api_client(settings: Settings | None = None) -> LLMClient:
    """Return the frontier API client for the hard-query route."""
    from sourcerer.llm.api_client import AnthropicClient

    settings = settings or get_settings()
    return AnthropicClient(
        api_key=settings.anthropic_api_key,
        base_url=settings.anthropic_base_url,
        model=settings.api_model,
    )


def client_for(route: str, settings: Settings | None = None) -> LLMClient:
    """Pick the generation backend for a router decision ("local" | "api")."""
    settings = settings or get_settings()
    if route == "api":
        return get_api_client(settings)
    return get_llm_client(settings)
