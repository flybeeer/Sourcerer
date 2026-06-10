"""Ollama backend for the LLM client wrapper.

Talks to a local Ollama server over HTTP:
- embeddings via POST /api/embed
- chat completions via POST /api/chat (non-streaming)
"""

from __future__ import annotations

import httpx

from sourcerer.llm.client import ChatResult

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

    def chat(self, messages: list[dict]) -> ChatResult:
        """Run a non-streaming chat completion and return the reply with usage.

        Ollama reports token counts as `prompt_eval_count` (input) and
        `eval_count` (output); local inference is self-hosted, so cost is $0.
        """
        resp = httpx.post(
            f"{self.base_url}/api/chat",
            json={"model": self.chat_model, "messages": messages, "stream": False},
            timeout=_CHAT_TIMEOUT,
        )
        resp.raise_for_status()
        body = resp.json()
        return ChatResult(
            text=body["message"]["content"],
            model=body.get("model", self.chat_model),
            input_tokens=body.get("prompt_eval_count", 0),
            output_tokens=body.get("eval_count", 0),
        )
