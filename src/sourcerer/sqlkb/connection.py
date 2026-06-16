"""Read-only access to the source SQL knowledge base.

Thin facade over the swappable backends (`backends.get_backend`): callers run a
validated SELECT against `connect_ro` without caring whether the source is SQLite
(the demo default) or DuckDB (analytics scale). Every connection is opened
read-only — a second line of defence behind the `safety` SQL validation.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sourcerer.config import Settings
from sourcerer.sqlkb.backends import get_backend


@contextmanager
def connect_ro(db_path: str | Path, settings: Settings | None = None) -> Iterator[Any]:
    """Yield a read-only connection to `db_path` using the configured backend.

    Raises FileNotFoundError with a clear message if the file is missing.
    """
    with get_backend(settings).connect_ro(db_path) as conn:
        yield conn
