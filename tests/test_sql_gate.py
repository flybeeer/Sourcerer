"""Tests for the SQL KB governance gate (Phase 10c, SQLGlot, no DB).

The catalog lookups are stubbed so these run without Postgres; they cover the
behaviours that matter for leakage: forbidden-table rejection, PII-column
masking (named and via SELECT *), and fail-closed handling of unparseable SQL
and ambiguous stars.
"""

import sqlglot

from sourcerer.config import Settings
from sourcerer.governance import sql_gate
from sourcerer.governance.catalog import Asset
from sourcerer.governance.principal import Principal

# A confidential table + a PII salary column on it.
_EMPLOYEES = Asset(kind="table", id="employees", classification="confidential", owner_team="hr")
_SALARY = Asset(
    kind="column",
    id="employees.salary",
    classification="confidential",
    owner_team="hr",
    pii_tags=["salary"],
)

_SCHEMA = {"employees": ["id", "name", "salary", "team"]}


def _settings(**kw) -> Settings:
    return Settings(sql_kb_backend="sqlite", governance_default_classification="public", **kw)


def _stub_catalog(monkeypatch, tables: list[Asset], columns: list[Asset]):
    """Stub catalog.get_asset / list_assets so the gate needs no database."""
    by_key = {(a.kind, a.id): a for a in (*tables, *columns)}
    monkeypatch.setattr(sql_gate.catalog, "get_asset", lambda kind, aid: by_key.get((kind, aid)))
    monkeypatch.setattr(
        sql_gate.catalog, "list_assets", lambda kind: [a for a in by_key.values() if a.kind == kind]
    )


def _norm(sql: str) -> str:
    return sqlglot.parse_one(sql, read="sqlite").sql()


# ---------- table-level rejection ----------


def test_forbidden_table_is_rejected(monkeypatch):
    _stub_catalog(monkeypatch, [_EMPLOYEES], [])
    alice = Principal(id="alice", clearance="internal", teams=["ops"])  # < confidential, not hr
    res = sql_gate.gate_sql("SELECT name FROM employees", alice, _settings(), schema=_SCHEMA)
    assert not res.allowed
    assert res.rejected_tables == ["employees"]


def test_untagged_table_is_allowed_by_default(monkeypatch):
    _stub_catalog(monkeypatch, [], [])  # nothing catalogued → default public
    anon = Principal(id="anon", clearance="public")
    res = sql_gate.gate_sql("SELECT count(*) FROM sales", anon, _settings(), schema={})
    assert res.allowed
    assert res.masked_columns == []


def test_authorized_principal_passes_clean(monkeypatch):
    _stub_catalog(monkeypatch, [_EMPLOYEES], [_SALARY])
    bob = Principal(id="bob", clearance="confidential", teams=["hr"])  # may read both
    res = sql_gate.gate_sql("SELECT salary FROM employees", bob, _settings(), schema=_SCHEMA)
    assert res.allowed
    assert res.masked_columns == []
    assert "***" not in res.sql


# ---------- column masking ----------


def test_named_pii_column_is_masked(monkeypatch):
    _stub_catalog(monkeypatch, [_EMPLOYEES], [_SALARY])
    # On the hr team (so the table is readable) but a junior who can't see salary.
    junior = Principal(id="dan", clearance="internal", teams=["hr"])
    # salary column is confidential/hr → readable by team. Use a principal NOT on hr
    # but who can read the table some other way: give table public, column confidential.
    res = sql_gate.gate_sql(
        "SELECT name, salary FROM employees", junior, _settings(), schema=_SCHEMA
    )
    # junior is on hr → can read salary too; expect NOT masked here.
    assert res.allowed
    assert res.masked_columns == []


def test_pii_column_masked_for_unauthorized_team(monkeypatch):
    public_table = Asset(kind="table", id="employees", classification="public")
    _stub_catalog(monkeypatch, [public_table], [_SALARY])
    # Can read the (public) table, but not the confidential/hr salary column.
    analyst = Principal(id="alice", clearance="internal", teams=["ops"])
    res = sql_gate.gate_sql(
        "SELECT name, salary FROM employees", analyst, _settings(), schema=_SCHEMA
    )
    assert res.allowed
    assert res.masked_columns == ["employees.salary"]
    assert _norm(res.sql) == _norm("SELECT name, '***' AS salary FROM employees")


def test_select_star_masks_pii_via_expansion(monkeypatch):
    public_table = Asset(kind="table", id="employees", classification="public")
    _stub_catalog(monkeypatch, [public_table], [_SALARY])
    analyst = Principal(id="alice", clearance="internal", teams=["ops"])
    res = sql_gate.gate_sql("SELECT * FROM employees", analyst, _settings(), schema=_SCHEMA)
    assert res.allowed
    assert res.masked_columns == ["employees.salary"]
    assert _norm(res.sql) == _norm("SELECT id, name, '***' AS salary, team FROM employees")


def test_select_star_without_schema_fails_closed(monkeypatch):
    public_table = Asset(kind="table", id="employees", classification="public")
    _stub_catalog(monkeypatch, [public_table], [_SALARY])
    analyst = Principal(id="alice", clearance="internal", teams=["ops"])
    res = sql_gate.gate_sql("SELECT * FROM employees", analyst, _settings(), schema=None)
    assert not res.allowed  # would expose salary via * with no schema to expand


# ---------- fail-closed parsing ----------


def test_unparseable_sql_fails_closed(monkeypatch):
    _stub_catalog(monkeypatch, [], [])
    res = sql_gate.gate_sql("NOT SQL AT ALL ))(", Principal(id="x"), _settings())
    assert not res.allowed
