"""Evaluate the Text-to-SQL path: execution accuracy vs ground-truth SQL.

For each question we run a hand-written reference SELECT to get the ground truth,
then have the model generate + run its own SELECT, and compare the result sets.
"Execution accuracy" = fraction of questions whose generated result matches the
reference result — the standard text-to-SQL metric, and what backs the README claim.

Alongside accuracy we report the operational numbers the later hardening exists to
move: end-to-end **latency**, SQL-generation **tokens**, how many tables schema
retrieval put in the prompt vs the whole schema (**prompt narrowing**), and how many
queries the cost **guards** aborted.

This makes real LLM calls (SQL generation), so it is not free. Build the demo DB
first if you don't have your own:

    python scripts/build_sql_demo.py
    python scripts/sql_eval.py
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from sourcerer.config import get_settings
from sourcerer.llm.client import get_api_client, get_llm_client
from sourcerer.sqlkb import text_to_sql
from sourcerer.sqlkb.connection import connect_ro
from sourcerer.sqlkb.schema import table_signatures

_TOL = 1e-6


def _run_reference(db_path: str, sql: str) -> list[tuple]:
    with connect_ro(db_path) as conn:
        return [tuple(r) for r in conn.execute(sql).fetchall()]


def _percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile (p in [0, 1]); 0.0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return ordered[int(k)]
    return ordered[lo] * (hi - k) + ordered[hi] * (k - lo)


def _cell_eq(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-4, abs_tol=_TOL)
    return str(a) == str(b)


def _results_match(expected: list[tuple], got: list[tuple]) -> bool:
    """Order-insensitive comparison of two result sets."""
    if len(expected) != len(got):
        return False
    remaining = list(got)
    for row in expected:
        match = next(
            (
                g
                for g in remaining
                if len(g) == len(row) and all(_cell_eq(a, b) for a, b in zip(row, g, strict=True))
            ),
            None,
        )
        if match is None:
            return False
        remaining.remove(match)
    return True


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Text-to-SQL execution-accuracy eval.")
    parser.add_argument(
        "--eval-set", default="eval/sql_eval_set.jsonl", help="JSONL of {question, reference_sql}."
    )
    parser.add_argument("--db", default=settings.sql_kb_path, help="SQLite DB to query.")
    args = parser.parse_args()

    if not Path(args.db).exists():
        raise SystemExit(f"No DB at {args.db}. Build one: python scripts/build_sql_demo.py")

    settings = settings.model_copy(update={"sql_kb_path": args.db})
    client = (
        get_api_client(settings)
        if settings.sql_kb_generate_with_api and settings.anthropic_api_key
        else get_llm_client(settings)
    )

    lines = Path(args.eval_set).read_text().splitlines()
    items = [json.loads(line) for line in lines if line.strip()]
    if not items:
        raise SystemExit(f"No questions in {args.eval_set}")
    total_tables = len(table_signatures(args.db, settings))
    print(f"Evaluating Text-to-SQL on {len(items)} questions (db: {args.db}) …\n")

    correct = 0
    latencies_ms: list[float] = []
    gen_tokens: list[int] = []
    prompt_tables: list[int] = []
    guard_trips = 0
    for item in items:
        question, ref_sql = item["question"], item["reference_sql"]
        expected = _run_reference(args.db, ref_sql)
        start = time.perf_counter()
        execution = text_to_sql.run(question, settings, client)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        gen_tokens.append(execution.input_tokens + execution.output_tokens)
        prompt_tables.append(execution.prompt_tables)
        if execution.error and "exceeded" in execution.error:
            guard_trips += 1

        ok = execution.ok and _results_match(expected, execution.rows)
        correct += ok
        mark = "✓" if ok else "✗"
        print(f"{mark} {question}  ({latencies_ms[-1]:.0f} ms)")
        print(f"    gen: {execution.sql if execution.ok else execution.error}")
        if not ok:
            print(f"    expected {expected} | got {execution.rows if execution.ok else '—'}")

    n = len(items)
    accuracy = correct / n if n else 0.0
    avg_tables = sum(prompt_tables) / n if n else 0.0
    print(f"\nExecution accuracy: {correct}/{n} = {accuracy:.0%}")
    print(
        "Latency ms: "
        f"mean {sum(latencies_ms) / n:.0f} · "
        f"median {_percentile(latencies_ms, 0.5):.0f} · "
        f"p95 {_percentile(latencies_ms, 0.95):.0f}"
    )
    print(f"SQL-gen tokens/query: mean {sum(gen_tokens) / n:.0f}")
    narrowing = f" ({avg_tables / total_tables:.0%} of schema)" if total_tables else ""
    print(f"Prompt schema: mean {avg_tables:.1f}/{total_tables} tables{narrowing}")
    print(f"Cost-guard aborts: {guard_trips}/{n}")


if __name__ == "__main__":
    main()
