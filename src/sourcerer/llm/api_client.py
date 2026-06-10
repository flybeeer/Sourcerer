"""Frontier API backend for the LLM client wrapper (the Phase 4 hard-query route).

Generation-only: embeddings stay on the local model so the hybrid theme holds.
Uses the official Anthropic SDK, pointed at `anthropic_base_url` when set so it
can run through a corporate gateway (e.g. the CodeSmart proxy) instead of the
public API. The SDK import is lazy so the base install (and the local-only path)
never needs the `anthropic` package.
"""

from __future__ import annotations

from sourcerer.llm.client import ChatResult

# Hard queries can warrant some reasoning; adaptive thinking lets the model decide.
# 4096 stays well under the SDK's non-streaming timeout guard.
_MAX_TOKENS = 4096


class AnthropicClient:
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(
            "The API client is generation-only; embeddings run on the local model."
        )

    def chat(self, messages: list[dict]) -> ChatResult:
        """Run a non-streaming completion via the Messages API and return usage.

        The Messages API takes the system prompt separately, so any system-role
        messages are pulled out of the list and concatenated into `system`.
        """
        import anthropic  # lazy: only needed when the API route is actually taken

        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set; the API route is unavailable.")

        client = anthropic.Anthropic(
            api_key=self.api_key,
            base_url=self.base_url or None,
        )

        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        convo = [m for m in messages if m["role"] != "system"]

        kwargs: dict = {
            "model": self.model,
            "max_tokens": _MAX_TOKENS,
            "thinking": {"type": "adaptive"},
            "messages": convo,
        }
        if system:
            kwargs["system"] = system

        resp = client.messages.create(**kwargs)
        text = "".join(block.text for block in resp.content if block.type == "text")
        return ChatResult(
            text=text,
            model=resp.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )
