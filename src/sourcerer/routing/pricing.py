"""Token pricing for cost estimation.

USD per 1,000,000 tokens, as (input, output). Source: Anthropic public pricing
(2026-06). Local / self-hosted models are absent from the table and priced at $0
— that is the whole point of routing easy traffic to the local model.

Gateway model names (e.g. `codesmart.claude.sonnet`) map to the underlying model's
public price, so cost numbers stay comparable to an Anthropic-direct baseline.
"""

from __future__ import annotations

PRICING: dict[str, tuple[float, float]] = {
    # Public Anthropic model IDs
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    # Gateway aliases (CodeSmart proxy) — priced as the underlying Sonnet
    "codesmart.claude.sonnet": (3.00, 15.00),
}


def price_for(model: str) -> tuple[float, float]:
    """Return (input, output) USD-per-1M for a model; $0 for unknown/local models."""
    return PRICING.get(model, (0.0, 0.0))


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate the USD cost of one call from its token usage."""
    price_in, price_out = price_for(model)
    return input_tokens / 1e6 * price_in + output_tokens / 1e6 * price_out


def is_priced(model: str) -> bool:
    """True if the model has a non-zero price (i.e. it's the paid API route)."""
    return model in PRICING
