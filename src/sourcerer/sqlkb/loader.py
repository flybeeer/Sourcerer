"""Path A: turn SQLite rows into documents for the RAG pipeline.

A user-written SELECT defines *what* to ingest; each returned row becomes one
document (1 row = 1 document), rendered as `column: value` lines. The `source`
label points back to the row so citations are traceable — e.g. `kb:42` (using an
id column) or `kb:row-7` (row ordinal when no id column is given).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sourcerer.sqlkb.connection import connect_ro
from sourcerer.sqlkb.safety import safe_select


def _row_text(row, columns: list[str]) -> str:
    """Render a row as `column: value` lines, skipping NULLs."""
    return "\n".join(f"{col}: {row[col]}" for col in columns if row[col] is not None)


def iter_rows(
    db_path: str | Path,
    query: str,
    *,
    id_col: str | None = None,
    source_prefix: str | None = None,
    max_rows: int = 100_000,
) -> Iterator[tuple[str, str]]:
    """Yield `(source_label, text)` for each row returned by `query`.

    Args:
        db_path: the SQLite file.
        query: a read-only SELECT choosing the rows/columns to ingest.
        id_col: a column whose value names each row's source (stable across re-ingest).
                Falls back to the row ordinal when not given.
        source_prefix: label namespace; defaults to the db file stem.
        max_rows: hard cap on rows pulled (ingestion safety valve).

    The query is validated read-only just like a generated one, so a bad SELECT
    can't mutate the source DB.
    """
    prefix = source_prefix or Path(db_path).stem
    validated = safe_select(query, max_rows)
    with connect_ro(db_path) as conn:
        cursor = conn.execute(validated)
        columns = [d[0] for d in cursor.description]
        for ordinal, row in enumerate(cursor, start=1):
            text = _row_text(row, columns)
            if not text.strip():
                continue
            key = row[id_col] if id_col and id_col in columns else f"row-{ordinal}"
            yield f"{prefix}:{key}", text
