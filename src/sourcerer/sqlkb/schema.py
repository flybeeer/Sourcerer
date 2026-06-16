"""Introspect a source schema into a compact prompt description.

The Text-to-SQL model needs to know the tables, columns, and a taste of the data
to write a correct query. We keep this small (a few sample rows per table) so it
fits the prompt without blowing up tokens on large databases. Table/column
introspection differs per engine, so it is delegated to the active backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sourcerer.config import Settings
from sourcerer.sqlkb.backends import SqlBackend, get_backend

_SAMPLE_ROWS = 3


def _table_signature(backend: SqlBackend, conn: Any, table: str) -> str:
    """A table's name + columns only — the cheap signature used to *rank* tables.

    No sample-row query, so this is safe to build for every table on a wide schema
    (the data peek is added later, only for the tables we actually keep).
    """
    cols = backend.list_columns(conn, table)
    col_defs = ", ".join(f"{c.name} {c.type or 'TEXT'}" for c in cols)
    return f"TABLE {table} ({col_defs})"


def _describe_table(backend: SqlBackend, conn: Any, table: str) -> str:
    lines = [_table_signature(backend, conn, table)]
    cols = backend.list_columns(conn, table)

    sample = conn.execute(f'SELECT * FROM "{table}" LIMIT {_SAMPLE_ROWS}').fetchall()
    if sample:
        names = [c.name for c in cols]
        lines.append("  sample rows:")
        for row in sample:
            values = ", ".join(f"{name}={row[i]!r}" for i, name in enumerate(names))
            lines.append(f"    {values}")
    return "\n".join(lines)


def table_signatures(db_path: str | Path, settings: Settings | None = None) -> dict[str, str]:
    """Return `{table: "TABLE x (cols)"}` for every user table (no data scan).

    The lightweight signal schema retrieval ranks against — cheap enough to build
    for hundreds of tables, unlike full descriptions with sample rows.
    """
    backend = get_backend(settings)
    with backend.connect_ro(db_path) as conn:
        return {t: _table_signature(backend, conn, t) for t in backend.list_tables(conn)}


def describe_schema(
    db_path: str | Path,
    settings: Settings | None = None,
    tables: list[str] | None = None,
) -> str:
    """Return a human/LLM-readable schema description for the prompt.

    Describes each table with its columns and a few sample rows. With `tables`,
    only those (in that order) are described — the subset schema retrieval keeps;
    otherwise every table. Raises if the database has no tables (nothing to query).
    """
    backend = get_backend(settings)
    with backend.connect_ro(db_path) as conn:
        available = backend.list_tables(conn)
        if not available:
            raise ValueError(f"SQL KB database has no tables: {db_path}")
        chosen = [t for t in tables if t in available] if tables is not None else available
        return "\n\n".join(_describe_table(backend, conn, t) for t in chosen)
