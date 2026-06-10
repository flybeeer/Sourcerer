"""Report on hybrid routing: local-vs-API split, cost per query, % saved.

Two sources:

    python scripts/route_report.py                 # aggregate the live query log
    python scripts/route_report.py --simulate      # route a built-in query set offline

`--simulate` needs no DB, no Ollama, and no API key: it runs the heuristic router
over a representative query set and prices each query from token *assumptions*
(documented below), so the split and savings math are reproducible for the README.
`--from-log` (default) aggregates whatever the running system has actually served.
"""

from __future__ import annotations

import argparse
import logging

from sourcerer.config import get_settings
from sourcerer.routing import router as query_router
from sourcerer.routing.pricing import estimate_cost
from sourcerer.routing.report import QueryRecord, render_report, summarize

# --- Offline simulation assumptions -----------------------------------------
# A real query's input is the retrieved context + question; output is the answer.
# These rough averages let the simulation produce honest *relative* cost numbers
# without running retrieval or generation. They are clearly labelled as estimates.
_ASSUMED_CONTEXT_TOKENS = 800  # ~5 retrieved chunks of context
_ASSUMED_ANSWER_TOKENS = 180
_TOKENS_PER_WORD = 1.3

# A representative mix: easy lookups, sensitive (→ local), and hard reasoning.
_SAMPLE_QUERIES = [
    "What is the company's PTO policy?",
    "Who is the head of the engineering department?",
    "When does the fiscal year end?",
    "Define the term 'vesting cliff' from the equity handbook.",
    "How many sick days do part-time employees get?",
    "List the steps to request a hardware upgrade.",
    "What is the employee salary band for a senior engineer?",
    "Summarize the confidential acquisition terms in the NDA.",
    "What are my social security and bank account details on file?",
    "Why did the migration to the new auth system reduce latency, and how does "
    "the token-refresh flow compare to the previous session-cookie approach?",
    "Compare the trade-offs between the on-prem and cloud deployment options and "
    "explain which one better fits our compliance requirements and why.",
    "Analyze the root cause of the Q3 revenue shortfall and evaluate the "
    "implications for next year's hiring plan.",
]


def _simulate(settings) -> list[QueryRecord]:
    """Route the sample queries and estimate each one's tokens + cost."""
    router = query_router.HeuristicRouter(
        local_model=settings.local_model,
        api_model=settings.api_model,
        hard_threshold=settings.router_hard_threshold,
    )
    records = []
    for q in _SAMPLE_QUERIES:
        decision = router.decide(q)
        in_tokens = _ASSUMED_CONTEXT_TOKENS + int(len(q.split()) * _TOKENS_PER_WORD)
        out_tokens = _ASSUMED_ANSWER_TOKENS
        cost = estimate_cost(decision.model, in_tokens, out_tokens)
        records.append(
            QueryRecord(
                route=decision.route,
                model=decision.model,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                cost_usd=cost,
            )
        )
        print(f"  [{decision.route:>5}] {q[:70]:<70}  — {decision.reason}")
    return records


def _from_log(limit: int) -> list[QueryRecord]:
    """Pull routed rows from the live query log."""
    from sourcerer.observability.logging import routed_queries

    return [
        QueryRecord(
            route=r["route"],
            model=r["model"],
            input_tokens=r["input_tokens"],
            output_tokens=r["output_tokens"],
            cost_usd=r["cost_usd"],
            latency_ms=r["latency_ms"] or 0,
        )
        for r in routed_queries(limit=limit)
    ]


def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Hybrid routing cost report.")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Route a built-in query set offline instead of reading the query log.",
    )
    parser.add_argument(
        "--limit", type=int, default=1000, help="Max log rows to aggregate (--from-log)."
    )
    args = parser.parse_args()

    if args.simulate:
        print(f"Simulating {len(_SAMPLE_QUERIES)} queries through the heuristic router:\n")
        records = _simulate(settings)
        print()
    else:
        records = _from_log(args.limit)
        if not records:
            print(
                "No routed queries in the log yet. Serve some via POST /query, "
                "or run with --simulate."
            )
            return

    summary = summarize(records, api_model=settings.api_model)
    print(render_report(summary, api_model=settings.api_model))


if __name__ == "__main__":
    main()
