"""Path A: turn SQLite rows into documents for the RAG pipeline.

A user-written SELECT defines *what* to ingest; each returned row becomes one
document (1 row = 1 document), rendered as `column: value` lines. The `source`
label points back to the row so citations are traceable — e.g. `kb:42` (using an
id column) or `kb:row-7` (row ordinal when no id column is given).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sourcerer.config import Settings
from sourcerer.sqlkb.connection import connect_ro
from sourcerer.sqlkb.safety import safe_select


def _row_text(row, columns: list[str]) -> str:
    """Render a row as `column: value` lines, skipping NULLs.

    Accesses values positionally so it works across backends (SQLite Row supports
    name lookup; DuckDB returns plain tuples — index access is the common surface).
    """
    return "\n".join(f"{col}: {row[i]}" for i, col in enumerate(columns) if row[i] is not None)


def iter_rows(
    db_path: str | Path,
    query: str,
    *,
    id_col: str | None = None,
    source_prefix: str | None = None,
    max_rows: int = 100_000,
    settings: Settings | None = None,
) -> Iterator[tuple[str, str]]:
    """Yield `(source_label, text)` for each row returned by `query`.

    Args:
        db_path: the source database file.
        query: a read-only SELECT choosing the rows/columns to ingest.
        id_col: a column whose value names each row's source (stable across re-ingest).
                Falls back to the row ordinal when not given.
        source_prefix: label namespace; defaults to the db file stem.
        max_rows: hard cap on rows pulled (ingestion safety valve).
        settings: selects the backend (defaults to the process settings).

    The query is validated read-only just like a generated one, so a bad SELECT
    can't mutate the source DB.
    """
    prefix = source_prefix or Path(db_path).stem
    validated = safe_select(query, max_rows)
    with connect_ro(db_path, settings) as conn:
        cursor = conn.execute(validated)
        columns = [d[0] for d in cursor.description]
        id_idx = columns.index(id_col) if id_col and id_col in columns else None
        for ordinal, row in enumerate(cursor.fetchall(), start=1):
            text = _row_text(row, columns)
            if not text.strip():
                continue
            key = row[id_idx] if id_idx is not None else f"row-{ordinal}"
            yield f"{prefix}:{key}", text
