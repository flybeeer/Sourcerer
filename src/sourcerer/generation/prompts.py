"""Prompt construction for grounded, cited answers.

The system prompt enforces the project's hard rule: answer only from the
provided sources, and say "I don't know" rather than guessing.
"""

from __future__ import annotations

from sourcerer.retrieval.types import RetrievedChunk

NO_ANSWER = "I don't know based on the provided documents."

_RULES = [
    "Use ONLY the information in the numbered sources below. Do not use outside knowledge.",
    "Cite the sources you rely on inline using bracketed numbers like [1] or [2].",
    f'If the sources do not contain the answer, reply with exactly: "{NO_ANSWER}"',
    "Be concise and factual.",
]

SYSTEM_PROMPT = (
    "You are Sourcerer, a knowledge assistant that answers strictly "
    "from a set of provided sources.\n\nRules:\n" + "\n".join(f"- {rule}" for rule in _RULES)
)


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as a numbered source list for the prompt."""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        header = f"[{i}] (source: {chunk.source}, chunk {chunk.chunk_index})"
        blocks.append(f"{header}\n{chunk.content}")
    return "\n\n".join(blocks)


def build_messages(query: str, chunks: list[RetrievedChunk]) -> list[dict]:
    """Build the chat messages for a grounded answer."""
    context = format_context(chunks)
    user_content = (
        f"Sources:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the sources above, with inline citations."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
