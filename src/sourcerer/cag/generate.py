"""Answer from a preloaded cache — no per-query retrieval (CAG PoC).

The whole authorized corpus is already in `cache.context`; the model answers from
it directly. Contrast with the RAG path (retrieval/retriever.py) which embeds the
query and searches per call. Governance was enforced when the cache was built, so
nothing here needs a per-query gate.
"""

from __future__ import annotations

from dataclasses import dataclass

from sourcerer.cag.cache import ContextCache
from sourcerer.llm.client import LLMClient

_SYSTEM = (
    "You answer strictly from the CONTEXT below. If the answer is not in the "
    "context, reply exactly: I don't know. Cite the [source: ...] you used."
)


@dataclass
class CagAnswer:
    text: str
    sources_in_context: list[str]
    model: str
    input_tokens: int
    output_tokens: int


def answer(question: str, cache: ContextCache, client: LLMClient) -> CagAnswer:
    """Answer `question` from the preloaded cache context (no retrieval)."""
    messages = [
        {"role": "system", "content": _SYSTEM},
        # Context first (stable prefix → KV-cache reuse), question last.
        {"role": "user", "content": f"CONTEXT:\n{cache.context}\n\nQUESTION: {question}"},
    ]
    res = client.chat(messages)
    return CagAnswer(
        text=res.text,
        sources_in_context=cache.sources,
        model=res.model,
        input_tokens=res.input_tokens,
        output_tokens=res.output_tokens,
    )
