"""Introspect a SQLite schema into a compact prompt description.

The Text-to-SQL model needs to know the tables, columns, and a taste of the data
to write a correct query. We keep this small (a few sample rows per table) so it
fits the prompt without blowing up tokens on large databases.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from sourcerer.sqlkb.connection import connect_ro

_SAMPLE_ROWS = 3


def _table_names(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows]


def _describe_table(conn: sqlite3.Connection, table: str) -> str:
    # PRAGMA table_info: (cid, name, type, notnull, dflt_value, pk).
    cols = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    col_defs = ", ".join(f"{c['name']} {c['type'] or 'TEXT'}" for c in cols)
    lines = [f"TABLE {table} ({col_defs})"]

    sample = conn.execute(f'SELECT * FROM "{table}" LIMIT {_SAMPLE_ROWS}').fetchall()
    if sample:
        col_names = [c["name"] for c in cols]
        lines.append("  sample rows:")
        for row in sample:
            values = ", ".join(f"{name}={row[name]!r}" for name in col_names)
            lines.append(f"    {values}")
    return "\n".join(lines)


def describe_schema(db_path: str | Path) -> str:
    """Return a human/LLM-readable schema description for the prompt.

    Lists every user table with its columns and up to a few sample rows. Raises
    if the database has no tables (nothing to query).
    """
    with connect_ro(db_path) as conn:
        tables = _table_names(conn)
        if not tables:
            raise ValueError(f"SQL KB database has no tables: {db_path}")
        return "\n\n".join(_describe_table(conn, t) for t in tables)
