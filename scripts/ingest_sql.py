"""CLI entrypoint for ingesting SQLite rows as documents (Phase 7, Path A).

Usage:
    python scripts/ingest_sql.py "SELECT id, region, notes FROM sales" \
        [--db data/kb.sqlite] [--id-col id] [--source-prefix sales]

Each row returned by the SELECT becomes one document (1 row = 1 document), then
goes through the normal chunk -> embed -> store pipeline into pgvector. Use this
for *content* questions; analytical questions are answered live via Text-to-SQL.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sourcerer.config import get_settings
from sourcerer.ingestion.pipeline import ingest_sql


def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Ingest SQLite rows into pgvector.")
    parser.add_argument("query", help="A read-only SELECT choosing the rows/columns to ingest.")
    parser.add_argument(
        "--db",
        default=settings.sql_kb_path,
        help=f"Path to the SQLite file (default: SQL_KB_PATH={settings.sql_kb_path}).",
    )
    parser.add_argument(
        "--id-col",
        default=None,
        help="Column whose value names each row's source (default: row ordinal).",
    )
    parser.add_argument(
        "--source-prefix",
        default=None,
        help="Source label namespace (default: the db file stem).",
    )
    args = parser.parse_args()

    if not Path(args.db).exists():
        raise SystemExit(f"Database does not exist: {args.db}")

    summary = ingest_sql(args.db, args.query, id_col=args.id_col, source_prefix=args.source_prefix)
    print(f"Done. Ingested {summary['documents']} row-document(s), {summary['chunks']} chunk(s).")


if __name__ == "__main__":
    main()
