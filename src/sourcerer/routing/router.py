"""Per-query routing: local model vs. frontier API model.

The router decides, for each query, which generation backend to use:

- **local** — easy / sensitive / high-volume queries. Cheap and private.
- **api** — hard / complex-reasoning queries that justify the frontier model.

`ROUTER_STRATEGY=heuristic` is implemented here: a transparent, rule-based scorer
that estimates a query's difficulty in [0, 1] and routes anything at or above
`router_hard_threshold` to the API. Privacy always wins — a query that trips a
sensitivity rule is kept local even if it scores as hard.

Every decision carries a human-readable `reason` and the raw `signals`, so the
route taken is inspectable (logged per query, and surfaced in the API response).
A `classifier` strategy can slot in later behind the same `route()` entry point.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sourcerer.config import Settings

# --- Heuristic vocabularies -------------------------------------------------

# Sensitivity → keep local for privacy, regardless of difficulty.
_SENSITIVE = (
    "salary",
    "compensation",
    "payroll",
    "ssn",
    "social security",
    "password",
    "passwd",
    "credential",
    "api key",
    "secret key",
    "private key",
    "confidential",
    "proprietary",
    "nda",
    "internal only",
    "internal-only",
    "personal data",
    "pii",
    "patient",
    "diagnosis",
    "medical record",
    "bank account",
    "termination",
    "layoff",
    "lawsuit",
)

# Complex-reasoning markers → push difficulty up.
_REASONING = (
    "why",
    "how does",
    "how do",
    "compare",
    "contrast",
    "explain",
    "difference between",
    "trade-off",
    "tradeoff",
    "trade off",
    "pros and cons",
    "analyze",
    "evaluate",
    "assess",
    "implication",
    "synthesize",
    "derive",
    "prove",
    "rationale",
    "root cause",
    "step by step",
    "reason about",
    "relationship between",
    "design",
    "architect",
)

# Factual-lookup markers → easy / high-volume; pull difficulty down.
_SIMPLE = (
    "what is",
    "what's",
    "who is",
    "who's",
    "when is",
    "when did",
    "where is",
    "define",
    "definition of",
    "list the",
    "how many",
    "how much",
)

_WORD_RE = re.compile(r"\w+")


@dataclass
class RouteDecision:
    """The outcome of routing one query — designed to be logged and inspected."""

    route: str  # "local" | "api"
    model: str  # the model id that will actually run
    difficulty: float  # estimated difficulty in [0, 1]
    reason: str  # one-line rationale for the chosen route
    signals: dict = field(default_factory=dict)  # raw signals behind the score

    def as_local_fallback(self, local_model: str, note: str) -> RouteDecision:
        """Return a copy rerouted to local (e.g. when the API is unavailable)."""
        return RouteDecision(
            route="local",
            model=local_model,
            difficulty=self.difficulty,
            reason=f"{self.reason}; {note}",
            signals=self.signals,
        )


class HeuristicRouter:
    """Rule-based difficulty scorer. Config-free so it's trivially testable."""

    def __init__(self, local_model: str, api_model: str, hard_threshold: float = 0.7) -> None:
        self.local_model = local_model
        self.api_model = api_model
        self.hard_threshold = hard_threshold

    def decide(self, query: str) -> RouteDecision:
        q = query.lower()

        # 1. Privacy gate — sensitive content stays local no matter how hard.
        sensitive_hits = [term for term in _SENSITIVE if term in q]
        difficulty, signals = self._difficulty(q)
        signals["sensitive_hits"] = sensitive_hits

        if sensitive_hits:
            return RouteDecision(
                route="local",
                model=self.local_model,
                difficulty=difficulty,
                reason=(
                    f"sensitive term {sensitive_hits[0]!r} → local route "
                    "(privacy override, never leaves the local model)"
                ),
                signals=signals,
            )

        # 2. Difficulty gate.
        if difficulty >= self.hard_threshold:
            drivers = signals["reasoning_hits"] or ["length"]
            return RouteDecision(
                route="api",
                model=self.api_model,
                difficulty=difficulty,
                reason=(
                    f"hard query (difficulty {difficulty:.2f} ≥ {self.hard_threshold:.2f}; "
                    f"drivers: {', '.join(drivers)}) → API route"
                ),
                signals=signals,
            )

        return RouteDecision(
            route="local",
            model=self.local_model,
            difficulty=difficulty,
            reason=(
                f"easy/high-volume query (difficulty {difficulty:.2f} < "
                f"{self.hard_threshold:.2f}) → local route"
            ),
            signals=signals,
        )

    @staticmethod
    def _difficulty(q: str) -> tuple[float, dict]:
        """Additive, capped difficulty scorer. Returns (score in [0,1], signals)."""
        n_words = len(_WORD_RE.findall(q))
        reasoning_hits = [kw for kw in _REASONING if kw in q]
        simple_hits = [kw for kw in _SIMPLE if kw in q]
        n_questions = q.count("?")

        score = 0.0
        # Complex-reasoning markers are the strongest signal: one marker is
        # suggestive, but several distinct ones (analyze + evaluate + compare …)
        # are enough to route on their own, even for a short query.
        if reasoning_hits:
            score += min(0.7, 0.4 + 0.15 * (len(reasoning_hits) - 1))
        # Long queries tend to be harder / multi-faceted.
        if n_words >= 40:
            score += 0.3
        elif n_words >= 20:
            score += 0.15
        # Multi-part questions ("... ? and ... ?") add a little.
        if n_questions >= 2:
            score += 0.15
        # A short, plainly factual lookup pulls back toward local.
        if simple_hits and n_words < 20 and not reasoning_hits:
            score -= 0.3

        score = max(0.0, min(1.0, score))
        signals = {
            "n_words": n_words,
            "n_questions": n_questions,
            "reasoning_hits": reasoning_hits,
            "simple_hits": simple_hits,
        }
        return round(score, 3), signals


def route(query: str, settings: Settings) -> RouteDecision:
    """Route one query using the configured strategy (default: heuristic)."""
    router = HeuristicRouter(
        local_model=settings.local_model,
        api_model=settings.api_model,
        hard_threshold=settings.router_hard_threshold,
    )
    # `classifier` is a future strategy; until it exists, heuristic is the path.
    return router.decide(query)
