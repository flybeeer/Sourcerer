"""Tests for the SQL knowledge base (Phase 7): safety, routing predicate, and
both paths against a temp SQLite fixture. No Postgres/Ollama needed — the LLM is
faked and Path A is tested at the loader level.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from sourcerer.config import Settings
from sourcerer.llm.client import ChatResult
from sourcerer.routing.router import is_analytical_query
from sourcerer.sqlkb import answer as sql_answer
from sourcerer.sqlkb import text_to_sql
from sourcerer.sqlkb.loader import iter_rows
from sourcerer.sqlkb.safety import UnsafeSQLError, safe_select
from sourcerer.sqlkb.schema import describe_schema


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
        "SELECT 1; DROP TABLE sales",
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
    "q", ["What is the return policy?", "Explain the onboarding process", "นโยบายคืนสินค้าเป็นยังไง"]
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
