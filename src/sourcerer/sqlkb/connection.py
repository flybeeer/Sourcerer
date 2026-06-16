"""Read-only access to the source SQLite database.

Every connection is opened with the `mode=ro` URI so writes are impossible at the
engine level — a second line of defence behind the `safety` SQL validation. The
source DB is external input; we only ever read from it.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def connect_ro(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    """Yield a read-only SQLite connection to `db_path`.

    Raises FileNotFoundError with a clear message if the file is missing (a
    `mode=ro` connect to a non-existent file otherwise fails opaquely).
    """
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"SQL KB database not found: {path} (set SQL_KB_PATH)")
    # The query string is the SQLite URI form: open the existing file read-only.
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        conn.close()
