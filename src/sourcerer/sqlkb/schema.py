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


def _describe_table(backend: SqlBackend, conn: Any, table: str) -> str:
    cols = backend.list_columns(conn, table)
    col_defs = ", ".join(f"{c.name} {c.type or 'TEXT'}" for c in cols)
    lines = [f"TABLE {table} ({col_defs})"]

    sample = conn.execute(f'SELECT * FROM "{table}" LIMIT {_SAMPLE_ROWS}').fetchall()
    if sample:
        names = [c.name for c in cols]
        lines.append("  sample rows:")
        for row in sample:
            values = ", ".join(f"{name}={row[i]!r}" for i, name in enumerate(names))
            lines.append(f"    {values}")
    return "\n".join(lines)


def describe_schema(db_path: str | Path, settings: Settings | None = None) -> str:
    """Return a human/LLM-readable schema description for the prompt.

    Lists every user table with its columns and up to a few sample rows. Raises
    if the database has no tables (nothing to query).
    """
    backend = get_backend(settings)
    with backend.connect_ro(db_path) as conn:
        tables = backend.list_tables(conn)
        if not tables:
            raise ValueError(f"SQL KB database has no tables: {db_path}")
        return "\n\n".join(_describe_table(backend, conn, t) for t in tables)
