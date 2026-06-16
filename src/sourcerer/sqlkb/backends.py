"""Swappable read-only SQL backends behind the SQL knowledge base.

The source database is external input we only ever read from. *Which engine* holds
it is a deployment choice — SQLite for the demo, DuckDB (columnar) once the source
grows to analytics scale — so we put both behind one interface, mirroring how
`LOCAL_BACKEND` makes the LLM backend swappable. Pick with `SQL_KB_BACKEND`.

Each backend knows three engine-specific things; everything else (running a
validated SELECT, rendering rows) is plain DB-API and lives in the callers:
  * how to open the database read-only,
  * how to introspect tables/columns for the Text-to-SQL prompt,
  * how to bound a runaway query (a cheap watchdog).
"""

from __future__ import annotations

import sqlite3
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sourcerer.config import Settings, get_settings

# How often (in engine VM ops) the SQLite progress handler fires to check the
# guards — small enough that a wall-clock timeout stays responsive, large enough
# that the callback overhead is negligible.
_PROGRESS_INTERVAL = 50_000


class QueryCostError(RuntimeError):
    """Raised when a generated query is aborted for exceeding a runtime budget."""


@dataclass(frozen=True)
class Column:
    """One column of a source table, for the schema description."""

    name: str
    type: str


@dataclass
class Guard:
    """Mutable handle a running query's watchdog flips when it trips a budget.

    The caller inspects `tripped` after the engine raises so a generic "query
    interrupted" becomes a precise "exceeded time limit / scan budget".
    """

    tripped: str | None = None


def _require_file(db_path: str | Path) -> Path:
    """Resolve `db_path`, failing loudly if it is missing.

    A read-only connect to a non-existent file otherwise fails opaquely (SQLite)
    or silently creates an empty DB (DuckDB), so we guard up front.
    """
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"SQL KB database not found: {path} (set SQL_KB_PATH)")
    return path


class SqlBackend(ABC):
    """Read-only access to one source database engine."""

    name: str

    @abstractmethod
    def connect_ro(self, db_path: str | Path) -> AbstractContextManager[Any]:
        """Yield a read-only DB-API connection to `db_path`."""

    @abstractmethod
    def list_tables(self, conn: Any) -> list[str]:
        """Return user table names (excluding engine-internal tables)."""

    @abstractmethod
    def list_columns(self, conn: Any, table: str) -> list[Column]:
        """Return `table`'s columns in definition (i.e. `SELECT *`) order."""

    @contextmanager
    def query_guard(self, conn: Any, settings: Settings) -> Iterator[Guard]:
        """Bound a query running on `conn` to the configured cost budgets.

        Wraps the `execute`/`fetchall` so a runaway query (e.g. a full scan on a
        big table) is aborted. The default applies no runtime guard — only the
        `LIMIT` that `safety.safe_select` already adds. Subclasses override.
        """
        yield Guard()


class SQLiteBackend(SqlBackend):
    """The default: a single SQLite file opened with the `mode=ro` URI."""

    name = "sqlite"

    @contextmanager
    def connect_ro(self, db_path: str | Path) -> Iterator[sqlite3.Connection]:
        path = _require_file(db_path)
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            conn.row_factory = sqlite3.Row
            yield conn
        finally:
            conn.close()

    def list_tables(self, conn: sqlite3.Connection) -> list[str]:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        return [r[0] for r in rows]

    def list_columns(self, conn: sqlite3.Connection, table: str) -> list[Column]:
        # PRAGMA table_info: (cid, name, type, notnull, dflt_value, pk).
        cols = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
        return [Column(c[1], c[2] or "TEXT") for c in cols]

    @contextmanager
    def query_guard(self, conn: sqlite3.Connection, settings: Settings) -> Iterator[Guard]:
        # SQLite has no statement timeout, but a progress handler fires every N VM
        # ops and aborts the query when it returns non-zero — so one callback can
        # enforce both a wall-clock deadline and a scan-work (op-count) budget.
        guard = Guard()
        timeout = settings.sql_kb_timeout_s
        deadline = time.monotonic() + timeout if timeout and timeout > 0 else None
        budget = settings.sql_kb_max_scan_ops
        max_calls = budget // _PROGRESS_INTERVAL if budget and budget > 0 else None
        calls = 0

        def handler() -> int:
            nonlocal calls
            calls += 1
            if deadline and time.monotonic() > deadline:
                guard.tripped = "time limit"
                return 1  # non-zero → SQLite aborts with OperationalError
            if max_calls and calls > max_calls:
                guard.tripped = "scan budget"
                return 1
            return 0

        conn.set_progress_handler(handler, _PROGRESS_INTERVAL)
        try:
            yield guard
        finally:
            conn.set_progress_handler(None, _PROGRESS_INTERVAL)


class DuckDBBackend(SqlBackend):
    """Columnar engine for analytics-scale sources (Parquet/`.duckdb` files).

    Same read-only contract as SQLite, but aggregations push down to a column store
    instead of scanning a row store — the point of moving here when a table is big.
    Introspection uses `information_schema` (SQLite has no such views).
    """

    name = "duckdb"

    @contextmanager
    def connect_ro(self, db_path: str | Path) -> Iterator[Any]:
        path = _require_file(db_path)
        try:
            import duckdb
        except ModuleNotFoundError as exc:  # pragma: no cover - import guard
            raise ModuleNotFoundError(
                "the DuckDB backend needs the 'duckdb' package: pip install '.[duckdb]'"
            ) from exc
        conn = duckdb.connect(database=str(path), read_only=True)
        try:
            yield conn
        finally:
            conn.close()

    def list_tables(self, conn: Any) -> list[str]:
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchall()
        return [r[0] for r in rows]

    def list_columns(self, conn: Any, table: str) -> list[Column]:
        rows = conn.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = ? ORDER BY ordinal_position",
            [table],
        ).fetchall()
        return [Column(r[0], r[1] or "TEXT") for r in rows]

    @contextmanager
    def query_guard(self, conn: Any, settings: Settings) -> Iterator[Guard]:
        # DuckDB has no per-statement timeout, but `interrupt()` cancels the query
        # running on the connection — so a watchdog thread enforces the wall-clock
        # cap. (Scan-op budgeting is SQLite-specific; here LIMIT + timeout apply.)
        guard = Guard()
        timeout = settings.sql_kb_timeout_s
        if not timeout or timeout <= 0:
            yield guard
            return

        import threading

        done = threading.Event()

        def watchdog() -> None:
            if not done.wait(timeout):  # timed out before the query finished
                guard.tripped = "time limit"
                conn.interrupt()

        thread = threading.Thread(target=watchdog, daemon=True)
        thread.start()
        try:
            yield guard
        finally:
            done.set()
            thread.join(timeout=1)


_BACKENDS: dict[str, type[SqlBackend]] = {
    "sqlite": SQLiteBackend,
    "duckdb": DuckDBBackend,
}


def get_backend(settings: Settings | None = None) -> SqlBackend:
    """Return the configured read-only backend (mirrors `get_llm_client`)."""
    settings = settings or get_settings()
    name = settings.sql_kb_backend.lower()
    try:
        return _BACKENDS[name]()
    except KeyError:
        raise ValueError(
            f"unknown SQL_KB_BACKEND {name!r}; expected one of {sorted(_BACKENDS)}"
        ) from None
