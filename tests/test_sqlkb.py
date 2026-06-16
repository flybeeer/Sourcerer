"""Tests for the SQL knowledge base (Phase 7): safety, routing predicate, and
both paths against a temp SQLite fixture. No Postgres/Ollama needed — the LLM is
faked and Path A is tested at the loader level.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

from sourcerer.config import Settings
from sourcerer.llm.client import ChatResult
from sourcerer.routing.router import is_analytical_query
from sourcerer.sqlkb import answer as sql_answer
from sourcerer.sqlkb import schema_retrieval, text_to_sql
from sourcerer.sqlkb.backends import DuckDBBackend, SQLiteBackend, get_backend
from sourcerer.sqlkb.loader import iter_rows
from sourcerer.sqlkb.safety import UnsafeSQLError, safe_select
from sourcerer.sqlkb.schema import describe_schema, table_signatures


@pytest.fixture
def sales_db(tmp_path: Path) -> str:
    path = tmp_path / "kb.sqlite"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE sales (id INTEGER PRIMARY KEY, region TEXT, amount REAL, notes TEXT)"
    )
    conn.executemany(
        "INSERT INTO sales (region, amount, notes) VALUES (?, ?, ?)",
        [
            ("north", 1000.0, "fast shipping"),
            ("south", 500.0, "late delivery"),
            ("north", 1500.0, None),
        ],
    )
    conn.commit()
    conn.close()
    return str(path)


class FakeClient:
    """An LLMClient stub that returns canned text (the SQL or the phrased answer)."""

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def embed(self, texts):  # pragma: no cover - unused here
        return [[0.0] for _ in texts]

    def chat(self, messages) -> ChatResult:
        return ChatResult(text=self.reply, model="fake", input_tokens=10, output_tokens=5)


# --- safety -----------------------------------------------------------------


def test_safe_select_appends_limit():
    assert safe_select("SELECT * FROM sales", 50) == "SELECT * FROM sales LIMIT 50"


def test_safe_select_tightens_loose_limit():
    assert safe_select("SELECT * FROM sales LIMIT 9999", 50).lower().endswith("limit 50")


def test_safe_select_keeps_tighter_limit():
    assert safe_select("SELECT * FROM sales LIMIT 5", 50) == "SELECT * FROM sales LIMIT 5"


def test_safe_select_strips_fences():
    assert safe_select("```sql\nSELECT 1\n```", 50) == "SELECT 1 LIMIT 50"


@pytest.mark.parametrize(
    "bad",
    [
        "DELETE FROM sales",
        "UPDATE sales SET amount = 0",
        "DROP TABLE sales",
        "INSERT INTO sales VALUES (1)",
        "PRAGMA table_info(sales)",
        "",
    ],
)
def test_safe_select_rejects_unsafe(bad):
    with pytest.raises(UnsafeSQLError):
        safe_select(bad, 50)


def test_safe_select_extracts_sql_from_prose_prefix():
    # Small models sometimes prepend a sentence; the SELECT is still extracted.
    assert safe_select("Here is the SQL: SELECT COUNT(*) FROM sales", 50) == (
        "SELECT COUNT(*) FROM sales LIMIT 50"
    )


def test_safe_select_drops_trailing_explanation():
    # Chatty models add prose after the query; keep only the first statement.
    raw = "SELECT product FROM sales;\n\nThis query returns every product sold."
    assert safe_select(raw, 50) == "SELECT product FROM sales LIMIT 50"


def test_safe_select_takes_first_statement_only():
    # A tacked-on second statement is dropped, not run (read-only conn besides).
    assert safe_select("SELECT 1; DROP TABLE sales", 50) == "SELECT 1 LIMIT 50"


def test_safe_select_allows_cte_and_column_named_like_keyword():
    # `created_at` / a WITH-CTE must not trip the keyword filter.
    assert safe_select("WITH t AS (SELECT created_at FROM sales) SELECT * FROM t", 50)


# --- routing predicate ------------------------------------------------------


@pytest.mark.parametrize(
    "q",
    [
        "What is the total sales last year?",
        "how many customers",
        "average order value",
        "ยอดขายรวมของปีที่แล้ว",
        "ลูกค้าภาคเหนือมีกี่คน",
        "ค่าเฉลี่ยต่อเดือน",
    ],
)
def test_is_analytical_true(q):
    assert is_analytical_query(q)


@pytest.mark.parametrize(
    "q",
    ["What is the return policy?", "Explain the onboarding process", "นโยบายคืนสินค้าเป็นยังไง"],
)
def test_is_analytical_false(q):
    assert not is_analytical_query(q)


# --- schema -----------------------------------------------------------------


def test_describe_schema_lists_table_and_columns(sales_db):
    desc = describe_schema(sales_db)
    assert "TABLE sales" in desc
    assert "region" in desc and "amount" in desc
    assert "sample rows" in desc


# --- Path A: loader ---------------------------------------------------------


def test_iter_rows_one_doc_per_row_with_id_label(sales_db):
    rows = list(iter_rows(sales_db, "SELECT id, region, amount, notes FROM sales", id_col="id"))
    assert len(rows) == 3
    sources = [s for s, _ in rows]
    assert sources == ["kb:1", "kb:2", "kb:3"]
    # NULL notes are skipped in the rendered text.
    assert "notes:" not in rows[2][1]
    assert "region: north" in rows[0][1]


def test_iter_rows_falls_back_to_ordinal(sales_db):
    rows = list(iter_rows(sales_db, "SELECT region FROM sales", source_prefix="s"))
    assert [s for s, _ in rows] == ["s:row-1", "s:row-2", "s:row-3"]


def test_iter_rows_rejects_non_select(sales_db):
    with pytest.raises(UnsafeSQLError):
        list(iter_rows(sales_db, "DELETE FROM sales"))


# --- Path B: text-to-sql + answer ------------------------------------------


def _settings(db: str) -> Settings:
    return Settings(sql_kb_path=db, sql_kb_max_rows=50)


def test_text_to_sql_runs_generated_query(sales_db):
    client = FakeClient("SELECT SUM(amount) FROM sales")
    execution = text_to_sql.run("total sales", _settings(sales_db), client)
    assert execution.ok
    assert execution.rows == [(3000.0,)]


def test_text_to_sql_rejects_unsafe_generation(sales_db):
    client = FakeClient("DELETE FROM sales")
    execution = text_to_sql.run("wipe it", _settings(sales_db), client)
    assert not execution.ok
    assert "unsafe" in execution.error


def test_answer_builds_citation_with_sql(sales_db):
    sql_client = FakeClient("SELECT COUNT(*) FROM sales")
    answer_client = FakeClient("There are 3 sales records.")
    result = sql_answer.answer("how many sales", _settings(sales_db), sql_client, answer_client)
    assert result.text == "There are 3 sales records."
    assert len(result.citations) == 1
    assert "SELECT COUNT(*)" in result.citations[0].snippet
    assert result.input_tokens == 20  # summed across both calls


# --- swappable backends -----------------------------------------------------


def test_get_backend_default_is_sqlite():
    assert isinstance(get_backend(Settings()), SQLiteBackend)


def test_get_backend_selects_duckdb():
    assert isinstance(get_backend(Settings(sql_kb_backend="duckdb")), DuckDBBackend)


def test_get_backend_rejects_unknown():
    with pytest.raises(ValueError, match="unknown SQL_KB_BACKEND"):
        get_backend(Settings(sql_kb_backend="oracle"))


@pytest.fixture
def duckdb_sales_db(tmp_path: Path) -> str:
    """The same sales table, materialized in a DuckDB file (skips if duckdb absent)."""
    duckdb = pytest.importorskip("duckdb")
    path = tmp_path / "kb.duckdb"
    conn = duckdb.connect(str(path))
    conn.execute("CREATE TABLE sales (id INTEGER, region TEXT, amount DOUBLE, notes TEXT)")
    conn.executemany(
        "INSERT INTO sales VALUES (?, ?, ?, ?)",
        [
            (1, "north", 1000.0, "fast shipping"),
            (2, "south", 500.0, "late delivery"),
            (3, "north", 1500.0, None),
        ],
    )
    conn.close()
    return str(path)


def _duck_settings(db: str) -> Settings:
    return Settings(sql_kb_path=db, sql_kb_backend="duckdb", sql_kb_max_rows=50)


def test_duckdb_describe_schema(duckdb_sales_db):
    desc = describe_schema(duckdb_sales_db, _duck_settings(duckdb_sales_db))
    assert "TABLE sales" in desc
    assert "region" in desc and "amount" in desc
    assert "sample rows" in desc


def test_duckdb_text_to_sql_runs_generated_query(duckdb_sales_db):
    client = FakeClient("SELECT SUM(amount) FROM sales")
    execution = text_to_sql.run("total sales", _duck_settings(duckdb_sales_db), client)
    assert execution.ok
    assert execution.rows == [(3000.0,)]


def test_duckdb_loader_one_doc_per_row(duckdb_sales_db):
    rows = list(
        iter_rows(
            duckdb_sales_db,
            "SELECT id, region, amount, notes FROM sales",
            id_col="id",
            source_prefix="kb",
            settings=_duck_settings(duckdb_sales_db),
        )
    )
    assert [s for s, _ in rows] == ["kb:1", "kb:2", "kb:3"]
    assert "region: north" in rows[0][1]
    assert "notes:" not in rows[2][1]  # NULL notes skipped


def test_duckdb_execution_failure_is_caught(duckdb_sales_db):
    # A valid-looking SELECT that references a missing column → no-answer, not a crash.
    client = FakeClient("SELECT nope FROM sales")
    execution = text_to_sql.run("bad", _duck_settings(duckdb_sales_db), client)
    assert not execution.ok
    assert "SQL execution failed" in execution.error


# --- cost guards (timeout / scan budget) ------------------------------------

# A recursive CTE that scans far more than any budget here — a stand-in for a
# generated query that would full-scan a huge table. LIMIT can't prune the work.
_HEAVY_SQL = (
    "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c WHERE x < 100000000) "
    "SELECT count(*) FROM c"
)


def test_sqlite_scan_budget_aborts_runaway_query(sales_db):
    # timeout off, tiny op budget → deterministic abort regardless of machine speed.
    settings = Settings(sql_kb_path=sales_db, sql_kb_timeout_s=0, sql_kb_max_scan_ops=100_000)
    execution = text_to_sql.run("count forever", settings, FakeClient(_HEAVY_SQL))
    assert not execution.ok
    assert "scan budget" in execution.error


def test_sqlite_timeout_aborts_runaway_query(sales_db):
    settings = Settings(sql_kb_path=sales_db, sql_kb_timeout_s=0.05, sql_kb_max_scan_ops=0)
    execution = text_to_sql.run("count forever", settings, FakeClient(_HEAVY_SQL))
    assert not execution.ok
    assert "time limit" in execution.error


def test_guard_does_not_trip_a_normal_query(sales_db):
    # A fast query under a generous budget runs to completion, guard untripped.
    settings = Settings(sql_kb_path=sales_db, sql_kb_timeout_s=5.0, sql_kb_max_scan_ops=1_000_000)
    execution = text_to_sql.run("total", settings, FakeClient("SELECT SUM(amount) FROM sales"))
    assert execution.ok
    assert execution.rows == [(3000.0,)]


def test_duckdb_timeout_aborts_runaway_query(duckdb_sales_db):
    settings = Settings(
        sql_kb_path=duckdb_sales_db,
        sql_kb_backend="duckdb",
        sql_kb_timeout_s=0.1,
        sql_kb_max_rows=50,
    )
    execution = text_to_sql.run("count forever", settings, FakeClient(_HEAVY_SQL))
    assert not execution.ok
    assert "time limit" in execution.error


# --- schema retrieval (wide schemas) ----------------------------------------

_WIDE_TABLES = {
    "employees": "id INTEGER, full_name TEXT, salary REAL",
    "customers": "id INTEGER, name TEXT, city TEXT",
    "products": "id INTEGER, title TEXT, price REAL",
    "orders": "id INTEGER, customer_id INTEGER, total REAL",
    "shipments": "id INTEGER, order_id INTEGER, carrier TEXT",
    "suppliers": "id INTEGER, name TEXT, country TEXT",
    "invoices": "id INTEGER, order_id INTEGER, amount REAL",
    "payments": "id INTEGER, invoice_id INTEGER, method TEXT",
    "regions": "id INTEGER, region_name TEXT",
    "warehouses": "id INTEGER, location TEXT, capacity INTEGER",
}


@pytest.fixture
def wide_db(tmp_path: Path) -> str:
    """A 10-table schema — past the default top_k so retrieval kicks in."""
    path = tmp_path / "wide.sqlite"
    conn = sqlite3.connect(path)
    for table, cols in _WIDE_TABLES.items():
        conn.execute(f"CREATE TABLE {table} ({cols})")
        conn.execute(f"INSERT INTO {table} DEFAULT VALUES")
    conn.commit()
    conn.close()
    return str(path)


class BowEmbed:
    """Deterministic bag-of-words embedder so cosine ≈ token overlap (no service)."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        toks = [re.findall(r"[a-z0-9]+", t.lower()) for t in texts]
        vocab = sorted({w for ts in toks for w in ts if len(w) >= 3})
        index = {w: i for i, w in enumerate(vocab)}
        out = []
        for ts in toks:
            vec = [0.0] * len(vocab)
            for w in ts:
                if w in index:
                    vec[index[w]] += 1.0
            out.append(vec)
        return out


def test_table_signatures_are_cheap_and_complete(wide_db):
    sigs = table_signatures(wide_db)
    assert set(sigs) == set(_WIDE_TABLES)
    assert sigs["employees"].startswith("TABLE employees (")
    assert "sample rows" not in sigs["employees"]  # signature = no data peek


def test_lexical_ranking_prefers_matching_table(wide_db):
    sigs = table_signatures(wide_db)
    ranked = schema_retrieval._lexical_ranking("how many employees and their salary", sigs)
    assert ranked[0] == "employees"


def test_select_schema_narrows_to_top_k(wide_db):
    settings = Settings(sql_kb_path=wide_db, sql_kb_schema_top_k=3)
    desc = schema_retrieval.select_schema("total payments by method", wide_db, settings)
    assert desc.count("TABLE ") == 3  # only the retrieved tables
    assert "TABLE payments" in desc
    assert "TABLE warehouses" not in desc  # irrelevant table dropped


def test_select_schema_describes_all_under_threshold(sales_db):
    # 1 table ≤ top_k → identical to plain describe_schema, no embedding needed.
    settings = Settings(sql_kb_path=sales_db, sql_kb_schema_top_k=8)
    assert schema_retrieval.select_schema("anything", sales_db, settings) == describe_schema(
        sales_db, settings
    )


def test_select_schema_top_k_zero_describes_all(wide_db):
    settings = Settings(sql_kb_path=wide_db, sql_kb_schema_top_k=0)
    desc = schema_retrieval.select_schema("how many employees", wide_db, settings)
    assert desc.count("TABLE ") == len(_WIDE_TABLES)


def test_embedding_ranking_works_on_its_own(wide_db):
    # Proves the embedding leg actually ranks (not silently None → lexical fallback).
    sigs = table_signatures(wide_db)
    ranked = schema_retrieval._embedding_ranking("carrier that shipped the order", sigs, BowEmbed())
    assert ranked is not None
    assert ranked[0] == "shipments"


def test_select_tables_fuses_embedding_signal(wide_db):
    settings = Settings(sql_kb_path=wide_db, sql_kb_schema_top_k=3)
    sigs = table_signatures(wide_db)
    tables = schema_retrieval.select_tables(
        "which carrier shipped the order", sigs, settings, BowEmbed()
    )
    assert "shipments" in tables  # carrier/order signal, via lexical + embedding RRF
    assert len(tables) == 3


def test_select_tables_falls_back_when_embeddings_fail(wide_db):
    class Boom:
        def embed(self, texts):
            raise RuntimeError("embedding backend down")

    settings = Settings(sql_kb_path=wide_db, sql_kb_schema_top_k=3)
    sigs = table_signatures(wide_db)
    tables = schema_retrieval.select_tables("employee salary report", sigs, settings, Boom())
    assert "employees" in tables  # lexical ranking still works → no crash
