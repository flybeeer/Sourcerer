"""Text chunking — two swappable strategies, selected by config.

- "fixed":    a transparent sliding window over whitespace-delimited words
              ("tokens" in config are approximated as words).
- "semantic": split into sentences, embed them, and start a new chunk where the
              similarity between consecutive sentences drops below a threshold
              (or the chunk reaches chunk_size words). Keeps related sentences
              together so retrieval gets more coherent context.

`chunk(text, settings)` dispatches on settings.chunk_strategy. Phase 3 will
compare the two with the eval harness.
"""

from __future__ import annotations

import re

from sourcerer.config import Settings

# Sentence boundary: end punctuation (Latin or Thai sara) or a blank line.
_SENTENCE_RE = re.compile(r"(?<=[.!?。?!])\s+|\n{2,}")


def chunk(text: str, settings: Settings) -> list[str]:
    """Chunk text using the strategy named in settings.chunk_strategy."""
    if settings.chunk_strategy == "semantic":
        return semantic_chunks(text, settings.chunk_size, settings.semantic_threshold)
    if settings.chunk_strategy == "fixed":
        return chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    raise ValueError(f"Unknown chunk strategy: {settings.chunk_strategy!r}")


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


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on punctuation / blank lines."""
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def _cosine(a: list[float], b: list[float]) -> float:
    import numpy as np

    va, vb = np.asarray(a), np.asarray(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(va @ vb / denom) if denom else 0.0


def semantic_chunks(text: str, max_words: int, threshold: float) -> list[str]:
    """Group consecutive sentences until topic shift or size cap.

    A new chunk starts when the cosine similarity between adjacent sentence
    embeddings falls below `threshold`, or adding the next sentence would exceed
    `max_words`. Falls back to whole-text behaviour when there is 0–1 sentence
    (e.g. text without sentence delimiters).
    """
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return [text.strip()] if text.strip() else []

    # Lazy import: only the semantic strategy needs embeddings at chunk time.
    from sourcerer.ingestion.embeddings import embed_texts

    embeddings = embed_texts(sentences)

    chunks: list[str] = []
    current = [sentences[0]]
    current_words = len(sentences[0].split())

    for i in range(1, len(sentences)):
        sim = _cosine(embeddings[i - 1], embeddings[i])
        next_words = len(sentences[i].split())
        if sim < threshold or current_words + next_words > max_words:
            chunks.append(" ".join(current))
            current = [sentences[i]]
            current_words = next_words
        else:
            current.append(sentences[i])
            current_words += next_words

    chunks.append(" ".join(current))
    return chunks
