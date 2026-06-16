"""Path B: question → read-only SELECT → executed result.

Generates SQL from the schema with the LLM wrapper, validates it through
`safety.safe_select`, and runs it on the read-only connection. The executed SQL
and its rows are returned for the answer step to synthesize and cite.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from sourcerer.config import Settings
from sourcerer.generation.prompts import build_sql_messages
from sourcerer.llm.client import LLMClient
from sourcerer.sqlkb.connection import connect_ro
from sourcerer.sqlkb.safety import UnsafeSQLError, safe_select
from sourcerer.sqlkb.schema import describe_schema

# Stop a runaway generated query from scanning forever (SQLite has no statement
# timeout; the progress handler fires every N virtual-machine ops).
_PROGRESS_OPS = 1_000_000


@dataclass
class SQLExecution:
    """The outcome of one Text-to-SQL attempt — designed to be logged and cited."""

    sql: str  # the validated SQL that ran (or the raw attempt, if it failed)
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    error: str | None = None  # set when generation/validation/execution failed
    # Token usage from the SQL-generation call (the answer call is counted separately).
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def ok(self) -> bool:
        return self.error is None


def render_table(columns: list[str], rows: list[tuple], max_rows: int = 50) -> str:
    """Render result rows as a small markdown-ish table for the answer prompt."""
    if not columns:
        return "(no columns)"
    header = " | ".join(columns)
    sep = " | ".join("---" for _ in columns)
    body = [" | ".join("" if v is None else str(v) for v in row) for row in rows[:max_rows]]
    if not body:
        return f"{header}\n{sep}\n(no rows)"
    return "\n".join([header, sep, *body])


def _execute(db_path: str | Path, sql: str) -> tuple[list[str], list[tuple]]:
    with connect_ro(db_path) as conn:
        conn.set_progress_handler(lambda: None, _PROGRESS_OPS)  # cheap watchdog hook
        cursor = conn.execute(sql)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        rows = [tuple(r) for r in cursor.fetchall()]
    return columns, rows


def run(query: str, settings: Settings, client: LLMClient) -> SQLExecution:
    """Generate, validate, and execute SQL for `query`. Never raises on bad SQL."""
    schema = describe_schema(settings.sql_kb_path)
    result = client.chat(build_sql_messages(query, schema))
    raw_sql = result.text.strip()

    base = SQLExecution(
        sql=raw_sql,
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )
    try:
        validated = safe_select(raw_sql, settings.sql_kb_max_rows)
    except UnsafeSQLError as exc:
        base.error = f"unsafe SQL rejected: {exc}"
        return base

    base.sql = validated
    try:
        base.columns, base.rows = _execute(settings.sql_kb_path, validated)
    except sqlite3.Error as exc:
        base.error = f"SQL execution failed: {exc}"
    return base
