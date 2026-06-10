"""Phase 4 hybrid routing: decide local vs. API per query, and report on it."""

from sourcerer.routing.pricing import estimate_cost, price_for
from sourcerer.routing.report import QueryRecord, render_report, summarize
from sourcerer.routing.router import HeuristicRouter, RouteDecision, route

__all__ = [
    "route",
    "RouteDecision",
    "HeuristicRouter",
    "estimate_cost",
    "price_for",
    "QueryRecord",
    "summarize",
    "render_report",
]
