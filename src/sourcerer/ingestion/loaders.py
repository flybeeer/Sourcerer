"""Document loaders. Phase 1 supports PDF and Markdown.

Each loader returns plain text; chunking happens downstream. The `source` we
store is the file name, which is what citations point back to.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt"}


def load_pdf(path: Path) -> str:
    """Extract text from a PDF, joining pages with blank lines."""
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_text(path: Path) -> str:
    """Read a Markdown / plain-text file as-is."""
    return path.read_text(encoding="utf-8").strip()


def load_document(path: Path) -> str:
    """Load a single document, dispatching on file suffix."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in {".md", ".markdown", ".txt"}:
        return load_text(path)
    raise ValueError(f"Unsupported file type: {path.suffix} ({path})")


def iter_documents(root: Path) -> Iterator[tuple[Path, str]]:
    """Yield (path, text) for every supported document under `root`.

    Skips empty files (e.g. .gitkeep placeholders) silently.
    """
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = load_document(path)
        if text:
            yield path, text
