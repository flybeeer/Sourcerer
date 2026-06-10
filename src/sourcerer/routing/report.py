"""Routing cost report: local-vs-API split, cost per query, % saved vs API-only.

Pure aggregation over per-query records, so it works the same whether the records
come from the live query log or from an offline simulation. The savings figure is
the honest comparison the project cares about: what every query *actually* cost
(local = $0) versus what it *would* have cost if every query had gone to the API.
"""

from __future__ import annotations

from dataclasses import dataclass

from sourcerer.routing.pricing import estimate_cost


@dataclass
class QueryRecord:
    """One answered query's routing + usage, the unit the report aggregates."""

    route: str  # "local" | "api"
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int = 0


def summarize(records: list[QueryRecord], api_model: str) -> dict:
    """Aggregate records into the headline routing metrics.

    The API-only baseline prices *every* query's tokens at `api_model` rates —
    including the ones that actually ran locally — so the saved % answers
    "what did routing buy us versus sending everything to the frontier model".
    """
    n = len(records)
    if n == 0:
        return {"n": 0}

    local = [r for r in records if r.route == "local"]
    api = [r for r in records if r.route == "api"]

    actual_cost = sum(r.cost_usd for r in records)
    baseline_cost = sum(estimate_cost(api_model, r.input_tokens, r.output_tokens) for r in records)
    saved = baseline_cost - actual_cost

    def _avg_latency(rows: list[QueryRecord]) -> float:
        return sum(r.latency_ms for r in rows) / len(rows) if rows else 0.0

    return {
        "n": n,
        "local_count": len(local),
        "api_count": len(api),
        "local_pct": len(local) / n * 100,
        "api_pct": len(api) / n * 100,
        "actual_cost": actual_cost,
        "baseline_cost": baseline_cost,
        "cost_per_query": actual_cost / n,
        "baseline_cost_per_query": baseline_cost / n,
        "saved": saved,
        "saved_pct": (saved / baseline_cost * 100) if baseline_cost else 0.0,
        "avg_latency_local_ms": _avg_latency(local),
        "avg_latency_api_ms": _avg_latency(api),
    }


def render_report(summary: dict, api_model: str) -> str:
    """Render a summary dict as a Markdown report."""
    if summary.get("n", 0) == 0:
        return "No routed queries to report."

    lines = [
        "## Hybrid routing report",
        "",
        f"- Queries routed: **{summary['n']}**",
        f"- Local: **{summary['local_count']}** "
        f"({summary['local_pct']:.0f}%)  ·  "
        f"API: **{summary['api_count']}** ({summary['api_pct']:.0f}%)",
        f"- Avg latency: local {summary['avg_latency_local_ms']:.0f} ms  ·  "
        f"API {summary['avg_latency_api_ms']:.0f} ms",
        "",
        f"- Cost per query (actual): **${summary['cost_per_query']:.5f}**",
        f"- Cost per query (API-only baseline, `{api_model}`): "
        f"**${summary['baseline_cost_per_query']:.5f}**",
        f"- Total actual cost: ${summary['actual_cost']:.4f}  ·  "
        f"API-only baseline: ${summary['baseline_cost']:.4f}",
        "",
        f"### 💰 Saved **{summary['saved_pct']:.0f}%** vs sending every query to the API",
        f"(${summary['saved']:.4f} saved over {summary['n']} queries)",
    ]
    return "\n".join(lines)
