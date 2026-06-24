"""SQL KB governance gate eval (Phase 10c) — the gate must leak nothing.

Runs a matrix of (principal, generated SQL) through the SQLGlot+policy gate and
checks, independently of the gate's own bookkeeping, that the *rewritten* SQL is
safe for that principal:

  table leak  — the principal may not read a table the query touches, yet the gate
                let it through (should have rejected).
  PII leak    — a column the principal may not read still appears raw in the
                rewritten projections (should have been masked to '***').

Reports leakage gated vs. an ungoverned baseline (no gate), plus reject/mask
counts and the p95 gate latency. The SQL is canned, so this runs offline (no LLM,
no Ollama) and is fully deterministic.

    python scripts/sql_gate_eval.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sqlglot
from sqlglot import exp

from sourcerer.config import get_settings
from sourcerer.governance import policy, sql_gate
from sourcerer.governance.catalog import Asset, init_catalog_schema, upsert_asset
from sourcerer.governance.principal import ANONYMOUS, Principal, get_principal, upsert_principal

# Scenario: a public table whose `salary` column is confidential PII, plus a
# restricted `board` table. This exercises BOTH gate behaviours — column masking
# (table readable, column not) and whole-table rejection.
TABLES = [
    Asset(kind="table", id="employees", classification="public"),
    Asset(kind="table", id="board", classification="restricted"),
]
COLUMNS = [
    Asset(
        kind="column",
        id="employees.salary",
        classification="confidential",
        owner_team="hr",
        pii_tags=["salary"],
    ),
]
SCHEMA = {"employees": ["id", "name", "salary", "team"], "board": ["id", "topic"]}

PRINCIPALS = [
    Principal(id="anonymous", clearance="public"),
    Principal(id="alice", roles=["analyst"], clearance="internal", teams=["ops"]),
    Principal(id="bob", roles=["hr"], clearance="confidential", teams=["hr"]),
    Principal(id="carol", roles=["admin"], clearance="restricted", teams=["hr"]),
]


def _seed() -> None:
    init_catalog_schema()
    for a in (*TABLES, *COLUMNS):
        upsert_asset(a)
    for p in PRINCIPALS:
        upsert_principal(p)


def _unreadable_tables(principal: Principal, sql: str) -> set[str]:
    parsed = sqlglot.parse_one(sql, read="sqlite")
    tables = {t.name for t in parsed.find_all(exp.Table)}
    out = set()
    for t in tables:
        asset = next(
            (a for a in TABLES if a.id == t), Asset(kind="table", id=t, classification="public")
        )
        if not policy.can_read(principal, asset):
            out.add(t)
    return out


def _raw_projected_columns(sql: str, single: str | None) -> set[str]:
    """Columns appearing raw in the (rewritten) projections, as table.col."""
    parsed = sqlglot.parse_one(sql, read="sqlite")
    cols = set()
    for proj in parsed.expressions:
        for col in proj.find_all(exp.Column):
            table = col.table or single
            if table:
                cols.add(f"{table}.{col.name}")
    return cols


def _unreadable_pii(principal: Principal) -> set[str]:
    return {a.id for a in COLUMNS if not policy.can_read(principal, a)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SQL KB governance gate eval.")
    parser.add_argument("--set", default="eval/sql_governance_eval_set.jsonl")
    args = parser.parse_args()

    settings = get_settings().model_copy(
        update={"governance_enabled": True, "governance_default_classification": "public"}
    )
    _seed()

    rows = [json.loads(x) for x in Path(args.set).read_text().splitlines() if x.strip()]
    n = len(rows)
    leak_gated = leak_base = rejected = masked_cols = 0
    latencies: list[float] = []

    print(f"{'principal':10} {'verdict':9} {'masked':18} sql")
    print("-" * 84)
    for r in rows:
        principal = get_principal(r["principal"]) or ANONYMOUS
        sql = r["sql"]
        single = next(
            iter({t.name for t in sqlglot.parse_one(sql, read="sqlite").find_all(exp.Table)}), None
        )

        bad_tables = _unreadable_tables(principal, sql)
        bad_pii = _unreadable_pii(principal)

        # Baseline (no gate): anything forbidden the SQL touches/exposes leaks.
        base_leaks = bool(bad_tables) or bool(bad_pii & _raw_projected_columns(sql, single))
        leak_base += base_leaks

        t0 = time.perf_counter()
        res = sql_gate.gate_sql(sql, principal, settings, schema=SCHEMA)
        latencies.append((time.perf_counter() - t0) * 1000)

        # Gated leak check, independent of the gate's own report.
        if not res.allowed:
            verdict = "REJECT"
            rejected += 1
            leaked = False
        else:
            still_raw = bad_pii & _raw_projected_columns(res.sql, single)
            leaked = bool(bad_tables) or bool(still_raw)
            verdict = "LEAK!" if leaked else ("MASK" if res.masked_columns else "allow")
            masked_cols += len(res.masked_columns)
        leak_gated += leaked

        print(f"{r['principal']:10} {verdict:9} {','.join(res.masked_columns) or '-':18} {sql}")

    latencies.sort()
    p95 = latencies[int(len(latencies) * 0.95) - 1] if latencies else 0.0
    print("-" * 84)
    print("\n== Headline ==")
    print(f"Cases:                  {n}")
    print(f"  leakage (baseline):   {leak_base}/{n} = {_pct(leak_base, n)}")
    print(
        f"  leakage (gated):      {leak_gated}/{n} = {_pct(leak_gated, n)}"
        f"   {'✅ 0 leaks' if leak_gated == 0 else '❌ LEAK'}"
    )
    print(f"  tables rejected:      {rejected}")
    print(f"  PII columns masked:   {masked_cols}")
    print(f"  gate latency p95:     {p95:.2f} ms")


def _pct(num: int, den: int) -> str:
    return f"{(100 * num / den):.0f}%" if den else "n/a"


if __name__ == "__main__":
    main()
