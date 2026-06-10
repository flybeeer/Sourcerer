"""vLLM backend for the LLM client wrapper (Phase 5, higher-throughput serving).

vLLM exposes an OpenAI-compatible HTTP API, so this talks to
`POST {VLLM_BASE_URL}/v1/chat/completions` and reads token usage from the
standard `usage` field. Generation-only: embeddings stay on Ollama (bge-m3),
so the local route can switch Ollama→vLLM without touching ingestion.

## Measuring throughput vs Ollama

Both backends sit behind the same `chat()` wrapper, so swapping is just
`LOCAL_BACKEND=ollama|vllm`. To compare, serve the *same* model on each and
drive concurrent load at the API:

    # 1. Ollama (default) — baseline
    LOCAL_BACKEND=ollama ; for i in $(seq 50); do curl -s localhost:8000/query \
        -d '{"query":"..."}' & done; wait

    # 2. vLLM — same model, OpenAI-compatible server on :8001
    python -m vllm.entrypoints.openai.api_server --model <model> --port 8001
    LOCAL_BACKEND=vllm VLLM_BASE_URL=http://localhost:8001 VLLM_MODEL=<model>

Watch tokens/sec and p50/p95 latency under N concurrent requests (the logged
`output_tokens` / `latency_ms` per query feed this). vLLM's continuous batching
typically wins decisively as concurrency rises; Ollama is simpler for single-user
dev. Report the throughput delta — that's the Phase 5 deliverable.
"""

from __future__ import annotations

import httpx

from sourcerer.llm.client import ChatResult

_CHAT_TIMEOUT = 300.0
_MAX_TOKENS = 1024


class VLLMClient:
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(
            "vLLM is generation-only here; embeddings run on the Ollama bge-m3 client."
        )

    def chat(self, messages: list[dict]) -> ChatResult:
        """Run a non-streaming chat completion via the OpenAI-compatible API."""
        if not self.base_url:
            raise RuntimeError("VLLM_BASE_URL is not set; the vLLM backend is unavailable.")
        resp = httpx.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "max_tokens": _MAX_TOKENS,
            },
            timeout=_CHAT_TIMEOUT,
        )
        resp.raise_for_status()
        body = resp.json()
        usage = body.get("usage", {})
        return ChatResult(
            text=body["choices"][0]["message"]["content"],
            model=body.get("model", self.model),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )
