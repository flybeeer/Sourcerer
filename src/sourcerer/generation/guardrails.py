"""Input/output guardrails (Phase 5).

Two guarantees:

1. **No relevant context → "I don't know".** The generator already refuses an
   empty context; `filter_relevant()` additionally drops chunks below an optional
   relevance floor so a confident-but-irrelevant top hit doesn't get answered.
2. **Basic prompt-injection screening.** `injection_reason()` flags obvious
   "ignore your instructions / reveal the system prompt" style inputs before they
   reach the model. Deliberately a simple, transparent denylist — not a complete
   defense, but it catches the common cases and is the right first layer.
"""

from __future__ import annotations

import re

from sourcerer.config import Settings
from sourcerer.retrieval.types import RetrievedChunk

# Patterns that strongly indicate an attempt to override the system prompt or
# exfiltrate it. Matched case-insensitively against the raw query.
_INJECTION_PATTERNS = [
    # `instruction` (singular stem) matches "instructions" too — re.search is a
    # substring match, so we don't need optional plurals.
    r"ignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier|preceding)\s+"
    r"(?:instruction|prompt|context|rule)",
    r"disregard\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|system)\s+"
    r"(?:instruction|prompt|rule)",
    r"forget\s+(?:all|everything|your)\s+(?:previous\s+|prior\s+)?(?:instruction|rule)",
    r"(?:reveal|show|print|repeat|leak)\s+(?:me\s+)?(?:your\s+|the\s+)?(?:system\s+)?"
    r"(?:prompt|instruction)",
    r"you\s+are\s+now\s+(?:a|an|in)\b",
    r"act\s+as\s+(?:if\s+you\s+are\s+|an?\s+)?(?:dan|jailbreak|developer\s+mode)",
    r"developer\s+mode",
    r"\bjailbreak\b",
    r"override\s+(?:your\s+|the\s+)?(?:safety\s+|system\s+)?(?:instruction|rule|setting)",
    r"ข้าม(?:คำสั่ง|กฎ)",  # Thai: skip the instructions/rules
    r"เพิกเฉย(?:ต่อ)?คำสั่ง",  # Thai: disregard the instructions
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


def injection_reason(query: str) -> str | None:
    """Return a short reason if the query looks like prompt injection, else None."""
    for pattern in _COMPILED:
        if pattern.search(query):
            return f"matched injection pattern: {pattern.pattern!r}"
    return None


def filter_relevant(chunks: list[RetrievedChunk], settings: Settings) -> list[RetrievedChunk]:
    """Drop chunks below the relevance floor (when configured).

    Only applied when a cross-encoder reranker ran (`reranker_type` local/cohere),
    because those scores are calibrated relevance values comparable across
    queries. Fusion/vector scores are not, so the floor is skipped for them. When
    `min_relevance_score` is None the input is returned unchanged — the
    empty-context guardrail in the generator is then the sole relevance gate.
    """
    floor = settings.min_relevance_score
    if floor is None or not chunks:
        return chunks
    if settings.reranker_type not in ("local", "cohere"):
        return chunks
    return [c for c in chunks if c.score >= floor]
