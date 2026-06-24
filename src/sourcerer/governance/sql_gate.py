"""SQL KB governance gate — table/column authorization on generated SQL (10c).

The Text-to-SQL path can't be gated by a source filter: the "query" is SQL the
LLM just wrote, and what it touches is locked inside that string. So the gate
*parses* it (SQLGlot — reliable where `safe_select`'s regex is not: `created_at`
never trips "create"), enriches each table/column with catalog attributes, and
asks the policy:

  - a forbidden TABLE  → reject the whole query  → "I don't know"
  - a PII COLUMN the principal can't read → SQLGlot **rewrites** the projection to
    '***' (mask), instead of dropping the whole answer

`safe_select` stays as the read-only guard *before* this; the gate adds the
AST-based policy layer on top (it does not replace it). The decision is delegated
to the same swappable PDP the document gate uses (`gate.get_pdp` — local
reference or Cerbos sidecar), so tables and columns are governed by one policy,
exactly like document sources.

Scope notes: masking applies to the final SELECT's projections (what's returned).
Unqualified columns are attributed when the query has a single table; `SELECT *`
is expanded from the schema so a PII column under a star is still masked (or, with
no schema available, the query is rejected rather than risk a leak).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from sourcerer.config import Settings
from sourcerer.governance import catalog
from sourcerer.governance import gate as gate_mod
from sourcerer.governance.catalog import Asset
from sourcerer.governance.principal import Principal

# SQL_KB_BACKEND values double as SQLGlot dialects.
_DIALECTS = {"sqlite": "sqlite", "duckdb": "duckdb"}

_MASK = "***"


@dataclass
class SqlGateResult:
    """Outcome of gating one generated SQL statement."""

    allowed: bool
    sql: str  # rewritten SQL (masked / star-expanded), or the original when clean
    rejected_tables: list[str] = field(default_factory=list)
    masked_columns: list[str] = field(default_factory=list)  # "table.col" masked
    reason: str | None = None


def _dialect(settings: Settings) -> str:
    return _DIALECTS.get(settings.sql_kb_backend.lower(), "")


def _table_asset(table: str, settings: Settings) -> Asset:
    """Catalogued table asset, or a default-classified one for untagged tables."""
    return catalog.get_asset("table", table) or Asset(
        kind="table", id=table, classification=settings.governance_default_classification
    )


def _resolve(
    principal: Principal, settings: Settings, tables: list[str]
) -> tuple[list[str], set[str]]:
    """Ask the PDP, in two batches, which tables/PII-columns are off-limits.

    Returns (rejected tables, masked "table.col" set). Routing through
    `gate.get_pdp` means the same decision point (local reference or Cerbos
    sidecar) governs the SQL path as governs document retrieval — one policy,
    three paths.
    """
    pdp = gate_mod.get_pdp(settings)

    table_assets = [_table_asset(t, settings) for t in tables]
    readable_tables = pdp.readable(principal, table_assets)
    rejected = [t for t in tables if t not in readable_tables]

    # Only catalogued columns of the touched tables are candidates for masking.
    table_set = set(tables)
    col_assets = [a for a in catalog.list_assets("column") if a.id.split(".", 1)[0] in table_set]
    readable_cols = pdp.readable(principal, col_assets) if col_assets else set()
    masked_set = {a.id for a in col_assets if a.id not in readable_cols}
    return rejected, masked_set


def gate_sql(
    sql: str,
    principal: Principal,
    settings: Settings,
    *,
    schema: dict[str, list[str]] | None = None,
) -> SqlGateResult:
    """Authorize generated SQL: reject forbidden tables, mask unreadable PII columns.

    `schema` maps table → ordered column names (used to expand `SELECT *` so masking
    still applies). Parse failures are treated as not-allowed (fail-closed).
    """
    dialect = _dialect(settings)
    try:
        parsed = sqlglot.parse_one(sql, read=dialect)
    except Exception as exc:  # unparseable → fail closed
        return SqlGateResult(allowed=False, sql=sql, reason=f"could not parse SQL: {exc}")

    tables = sorted({t.name for t in parsed.find_all(exp.Table)})
    rejected, masked_set = _resolve(principal, settings, tables)

    # 1) Table-level authorization — any forbidden table rejects the whole query.
    if rejected:
        return SqlGateResult(
            allowed=False,
            sql=sql,
            rejected_tables=rejected,
            reason=f"principal '{principal.id}' may not read table(s): {', '.join(rejected)}",
        )

    # 2) Column-level PII masking on the final SELECT's projections.
    single = tables[0] if len(tables) == 1 else None
    masked: list[str] = []
    new_projections: list[exp.Expression] = []
    changed = False

    for proj in parsed.expressions:
        if _is_star_projection(proj):
            expanded = _expand_star(proj, single, schema, masked_set, masked)
            if expanded is None:  # star with a PII column but no schema → fail closed
                return SqlGateResult(
                    allowed=False,
                    sql=sql,
                    reason="SELECT * may expose a masked column; name columns explicitly",
                )
            new_projections.extend(expanded)
            if not (len(expanded) == 1 and expanded[0] is proj):
                changed = True  # star was expanded (and possibly masked)
            continue

        if _projection_touches_masked(proj, single, masked_set, masked):
            new_projections.append(_masked_alias(proj.alias_or_name or _MASK))
            changed = True
        else:
            new_projections.append(proj)

    if changed:
        parsed.set("expressions", new_projections)

    return SqlGateResult(
        allowed=True,
        sql=parsed.sql(dialect) if changed else sql,
        masked_columns=masked,
    )


def _is_star_projection(proj: exp.Expression) -> bool:
    """True only for a `*` or `t.*` projection — not `count(*)` and the like."""
    return isinstance(proj, exp.Star) or (
        isinstance(proj, exp.Column) and isinstance(proj.this, exp.Star)
    )


def _masked_alias(name: str) -> exp.Expression:
    """`'***' AS <name>` — a projection that reveals nothing."""
    return exp.alias_(exp.Literal.string(_MASK), name)


def _projection_touches_masked(
    proj: exp.Expression, single: str | None, masked_set: set[str], masked: list[str]
) -> bool:
    """True if a projection references any column the principal must not see."""
    hit = False
    for col in proj.find_all(exp.Column):
        table = col.table or single
        if table and f"{table}.{col.name}" in masked_set:
            masked.append(f"{table}.{col.name}")
            hit = True
    return hit


def _expand_star(
    proj: exp.Expression,
    single: str | None,
    schema: dict[str, list[str]] | None,
    masked_set: set[str],
    masked: list[str],
) -> list[exp.Expression] | None:
    """Expand `*` / `t.*` into explicit (masked where needed) column projections.

    Returns None when a PII column would be exposed but the star can't be safely
    expanded (multi-table star, or no schema) — the caller turns that into a
    fail-closed rejection. When nothing under the star is masked, the star is left
    untouched.
    """
    table = (proj.table if isinstance(proj, exp.Column) else None) or single
    if table is None:
        # Multi-table `*`: can't attribute columns → can't prove it's leak-free.
        return None

    masked_here = {mid.split(".", 1)[1] for mid in masked_set if mid.startswith(f"{table}.")}
    cols = schema.get(table) if schema else None

    if cols is None:
        # No schema to expand with: safe only if the table has no masked columns.
        return [proj] if not masked_here else None

    out: list[exp.Expression] = []
    for c in cols:
        if c in masked_here:
            masked.append(f"{table}.{c}")
            out.append(_masked_alias(c))
        else:
            out.append(exp.column(c))
    return out
