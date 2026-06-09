"""Tests for the word-window chunker (pure logic, no I/O)."""

import pytest

from sourcerer.ingestion.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("", chunk_size=10, overlap=2) == []
    assert chunk_text("   ", chunk_size=10, overlap=2) == []


def test_short_text_is_single_chunk():
    assert chunk_text("one two three", chunk_size=10, overlap=2) == ["one two three"]


def test_window_sizes_and_overlap():
    words = " ".join(str(i) for i in range(1000))
    chunks = chunk_text(words, chunk_size=512, overlap=64)
    # all but the last chunk are full-size
    assert [len(c.split()) for c in chunks[:-1]] == [512] * (len(chunks) - 1)
    # consecutive chunks share `overlap` words
    first, second = chunks[0].split(), chunks[1].split()
    assert first[-64:] == second[:64]


def test_no_duplicate_tail_chunk():
    # 512 words, size 512 -> exactly one chunk, no empty trailing window
    words = " ".join(str(i) for i in range(512))
    assert len(chunk_text(words, chunk_size=512, overlap=64)) == 1


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, 11), (5, -1)])
def test_invalid_params_raise(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=size, overlap=overlap)
