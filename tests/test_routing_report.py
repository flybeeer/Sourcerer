"""Tests for routing cost aggregation (pure logic, no I/O)."""

from sourcerer.routing.pricing import estimate_cost
from sourcerer.routing.report import QueryRecord, summarize

_API_MODEL = "claude-sonnet-4-6"  # $3 / $15 per 1M tokens


def _local(in_tok=1000, out_tok=100):
    return QueryRecord("local", "local-llm", in_tok, out_tok, cost_usd=0.0, latency_ms=200)


def _api(in_tok=1000, out_tok=100):
    cost = estimate_cost(_API_MODEL, in_tok, out_tok)
    return QueryRecord("api", _API_MODEL, in_tok, out_tok, cost_usd=cost, latency_ms=900)


def test_empty_records():
    assert summarize([], _API_MODEL) == {"n": 0}


def test_split_counts_and_pct():
    s = summarize([_local(), _local(), _local(), _api()], _API_MODEL)
    assert s["n"] == 4
    assert s["local_count"] == 3
    assert s["api_count"] == 1
    assert s["local_pct"] == 75.0
    assert s["api_pct"] == 25.0


def test_all_local_saves_everything():
    s = summarize([_local(), _local()], _API_MODEL)
    assert s["actual_cost"] == 0.0
    assert s["baseline_cost"] > 0.0
    assert s["saved_pct"] == 100.0


def test_all_api_saves_nothing():
    s = summarize([_api(), _api()], _API_MODEL)
    # Actual == baseline when everything already went to the API.
    assert abs(s["saved"]) < 1e-12
    assert abs(s["saved_pct"]) < 1e-9


def test_saved_math_matches_components():
    records = [_local(2000, 200), _local(1000, 150), _api(1500, 300)]
    s = summarize(records, _API_MODEL)
    baseline = sum(estimate_cost(_API_MODEL, r.input_tokens, r.output_tokens) for r in records)
    actual = sum(r.cost_usd for r in records)
    assert abs(s["baseline_cost"] - baseline) < 1e-12
    assert abs(s["actual_cost"] - actual) < 1e-12
    assert abs(s["saved"] - (baseline - actual)) < 1e-12


def test_per_query_costs():
    s = summarize([_local(), _api()], _API_MODEL)
    assert abs(s["cost_per_query"] - s["actual_cost"] / 2) < 1e-12
    assert abs(s["baseline_cost_per_query"] - s["baseline_cost"] / 2) < 1e-12


def test_latency_by_route():
    s = summarize([_local(), _api()], _API_MODEL)
    assert s["avg_latency_local_ms"] == 200
    assert s["avg_latency_api_ms"] == 900
