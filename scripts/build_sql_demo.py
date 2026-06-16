"""Build a tiny demo SQLite database for the SQL knowledge base (Phase 7).

Creates a `sales` table with deterministic rows so the Text-to-SQL path and its
eval (scripts/sql_eval.py) have something concrete to run against. Safe to re-run:
it drops and recreates the table.

    python scripts/build_sql_demo.py            # writes to SQL_KB_PATH
    python scripts/build_sql_demo.py --db x.db  # custom path
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from sourcerer.config import get_settings

# (region, product, amount, sold_at) — two years so "last year" questions have a
# clear, checkable answer. Kept small and obvious on purpose.
_ROWS = [
    ("north", "widget", 1200.0, "2024-02-11"),
    ("north", "gadget", 800.0, "2024-05-03"),
    ("south", "widget", 1500.0, "2024-07-19"),
    ("south", "gizmo", 600.0, "2024-11-30"),
    ("central", "widget", 900.0, "2024-09-09"),
    ("north", "widget", 2000.0, "2025-01-15"),
    ("north", "gadget", 500.0, "2025-03-22"),
    ("south", "gizmo", 1100.0, "2025-06-01"),
    ("central", "gadget", 700.0, "2025-08-08"),
    ("central", "widget", 1300.0, "2025-10-10"),
]


def build(db_path: str | Path) -> int:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("DROP TABLE IF EXISTS sales")
        conn.execute(
            "CREATE TABLE sales ("
            "id INTEGER PRIMARY KEY, region TEXT, product TEXT, "
            "amount REAL, sold_at TEXT)"
        )
        conn.executemany(
            "INSERT INTO sales (region, product, amount, sold_at) VALUES (?, ?, ?, ?)",
            _ROWS,
        )
        conn.commit()
    finally:
        conn.close()
    return len(_ROWS)


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build the demo SQLite KB.")
    parser.add_argument("--db", default=settings.sql_kb_path, help="Output SQLite path.")
    args = parser.parse_args()
    n = build(args.db)
    print(f"Wrote {n} rows to {args.db}.")


if __name__ == "__main__":
    main()
