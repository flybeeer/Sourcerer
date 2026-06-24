"""Path B: question → read-only SELECT → executed result.

Generates SQL from the schema with the LLM wrapper, validates it through
`safety.safe_select`, and runs it on the read-only connection. The executed SQL
and its rows are returned for the answer step to synthesize and cite.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sourcerer.config import Settings
from sourcerer.generation.prompts import build_sql_messages
from sourcerer.governance.principal import Principal
from sourcerer.llm.client import LLMClient
from sourcerer.sqlkb.backends import QueryCostError, get_backend
from sourcerer.sqlkb.safety import UnsafeSQLError, safe_select
from sourcerer.sqlkb.schema_retrieval import select_schema


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
    # How many tables were described in the prompt — equals the whole schema unless
    # schema retrieval narrowed it (so eval can measure that narrowing per query).
    prompt_tables: int = 0
    # Governance gate (Phase 10c): whether the gate rejected the query, and which
    # PII columns it masked ("table.col"). Empty/False when governance is off.
    gate_rejected: bool = False
    masked_columns: list[str] = field(default_factory=list)

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


def _execute(settings: Settings, sql: str) -> tuple[list[str], list[tuple]]:
    backend = get_backend(settings)
    with (
        backend.connect_ro(settings.sql_kb_path) as conn,
        backend.query_guard(conn, settings) as guard,
    ):
        try:
            cursor = conn.execute(sql)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            rows = [tuple(r) for r in cursor.fetchall()]
        except Exception as exc:
            # A budget abort surfaces as a generic engine error; name the cause.
            if guard.tripped:
                raise QueryCostError(f"query exceeded {guard.tripped}") from exc
            raise
    return columns, rows


def _schema_map(settings: Settings) -> dict[str, list[str]]:
    """{table: [column names]} for the source DB — used to expand `SELECT *` safely."""
    backend = get_backend(settings)
    with backend.connect_ro(settings.sql_kb_path) as conn:
        return {
            t: [c.name for c in backend.list_columns(conn, t)] for t in backend.list_tables(conn)
        }


def run(
    query: str,
    settings: Settings,
    client: LLMClient,
    principal: Principal | None = None,
) -> SQLExecution:
    """Generate, validate, gate, and execute SQL for `query`. Never raises on bad SQL.

    When governance is enabled and a principal is given, the validated SQL passes
    through the SQLGlot+policy gate (Phase 10c): a forbidden table rejects the
    query; unreadable PII columns are masked before execution.
    """
    schema = select_schema(query, settings.sql_kb_path, settings)
    result = client.chat(build_sql_messages(query, schema))
    raw_sql = result.text.strip()

    base = SQLExecution(
        sql=raw_sql,
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        prompt_tables=sum(1 for line in schema.splitlines() if line.startswith("TABLE ")),
    )
    try:
        validated = safe_select(raw_sql, settings.sql_kb_max_rows)
    except UnsafeSQLError as exc:
        base.error = f"unsafe SQL rejected: {exc}"
        return base

    if settings.governance_enabled and principal is not None:
        # Lazy import: SQLGlot is the [governance] extra, only needed when gating.
        from sourcerer.governance import sql_gate

        decision = sql_gate.gate_sql(validated, principal, settings, schema=_schema_map(settings))
        if not decision.allowed:
            base.error = f"governance: {decision.reason}"
            base.gate_rejected = True
            return base
        validated = decision.sql
        base.masked_columns = decision.masked_columns

    base.sql = validated
    try:
        base.columns, base.rows = _execute(settings, validated)
    except Exception as exc:  # noqa: BLE001 - any engine error → guardrail "I don't know"
        # Each backend raises its own error type (sqlite3.Error, duckdb.Error, …);
        # a failed query is never fatal — it becomes a no-answer downstream.
        base.error = f"SQL execution failed: {exc}"
    return base
