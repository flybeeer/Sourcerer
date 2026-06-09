"""Ollama backend for the LLM client wrapper.

Talks to a local Ollama server over HTTP:
- embeddings via POST /api/embed
- chat completions via POST /api/chat (non-streaming)
"""

from __future__ import annotations

import httpx

# Embeddings are quick; generation on a 32B local model is not.
_EMBED_TIMEOUT = 60.0
_CHAT_TIMEOUT = 300.0


class OllamaClient:
    def __init__(self, base_url: str, chat_model: str, embedding_model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embedding_model = embedding_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per input, in order."""
        if not texts:
            return []
        resp = httpx.post(
            f"{self.base_url}/api/embed",
            json={"model": self.embedding_model, "input": texts},
            timeout=_EMBED_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]

    def chat(self, messages: list[dict]) -> str:
        """Run a non-streaming chat completion and return the reply text."""
        resp = httpx.post(
            f"{self.base_url}/api/chat",
            json={"model": self.chat_model, "messages": messages, "stream": False},
            timeout=_CHAT_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
