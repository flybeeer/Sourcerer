"""Generation: turn retrieved chunks into a cited answer.

This module owns only the "context -> answer" step. Retrieval is done by the
caller (the API route) and passed in, keeping retrieval and generation separate.
"""

from __future__ import annotations

from dataclasses import dataclass

from sourcerer.generation.prompts import NO_ANSWER, build_messages
from sourcerer.llm.client import get_llm_client
from sourcerer.retrieval.types import RetrievedChunk

_SNIPPET_CHARS = 240


@dataclass
class Citation:
    """A source the answer is grounded in, numbered to match inline markers."""

    n: int  # the [n] used in the answer text
    source: str
    chunk_index: int
    score: float
    snippet: str


@dataclass
class Answer:
    text: str
    citations: list[Citation]


def _snippet(content: str) -> str:
    return content if len(content) <= _SNIPPET_CHARS else content[:_SNIPPET_CHARS] + "…"


def generate(query: str, chunks: list[RetrievedChunk]) -> Answer:
    """Generate a grounded answer from the retrieved chunks.

    If nothing was retrieved, return the "I don't know" response without calling
    the model (guardrail for the empty-context case).
    """
    if not chunks:
        return Answer(text=NO_ANSWER, citations=[])

    messages = build_messages(query, chunks)
    text = get_llm_client().chat(messages)

    citations = [
        Citation(
            n=i,
            source=chunk.source,
            chunk_index=chunk.chunk_index,
            score=chunk.score,
            snippet=_snippet(chunk.content),
        )
        for i, chunk in enumerate(chunks, start=1)
    ]
    return Answer(text=text, citations=citations)
