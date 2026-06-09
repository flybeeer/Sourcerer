"""CLI entrypoint for ingestion.

Usage:
    python scripts/ingest.py [path]

`path` defaults to ./data/raw. Loads every supported document under it, chunks,
embeds, and stores the vectors in pgvector.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sourcerer.config import get_settings
from sourcerer.ingestion.pipeline import ingest_directory


def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Ingest documents into pgvector.")
    parser.add_argument(
        "path",
        nargs="?",
        default="data/raw",
        help="Directory of documents to ingest (default: data/raw).",
    )
    args = parser.parse_args()

    root = Path(args.path)
    if not root.exists():
        raise SystemExit(f"Path does not exist: {root}")

    summary = ingest_directory(root)
    print(f"Done. Ingested {summary['documents']} document(s), {summary['chunks']} chunk(s).")


if __name__ == "__main__":
    main()
