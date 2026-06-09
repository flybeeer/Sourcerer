"""Text chunking.

Phase 1 uses a simple, transparent sliding window over whitespace-delimited
words. "Tokens" in the config (chunk_size / chunk_overlap) are approximated as
words here — good enough to get the pipeline working end-to-end. Phase 2 will
experiment with smarter (semantic / contextual) strategies and a real tokenizer.
"""

from __future__ import annotations


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping word windows.

    Args:
        text: the full document text.
        chunk_size: target number of words per chunk.
        overlap: number of words shared between consecutive chunks.

    Returns:
        A list of chunk strings (empty list for empty input).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if window:
            chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks
