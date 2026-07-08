"""Governance leakage eval (Phase 10b) — the gate must leak nothing.

Runs an eval matrix of (principal, query, answer_source, authorized) and reports,
gated vs. an ungoverned baseline:

  leakage rate     — fraction of UNAUTHORIZED cases where the forbidden source
                     still reached the retrieved set. Headline: must be 0.
  recall@k         — fraction of AUTHORIZED cases where the right source was
                     retrieved. Governance must NOT hurt authorized recall
                     (the over-restriction / false-denial check).

Retrieval uses the BM25/keyword leg so the eval runs offline (no embedding
service). The gate's `source = ANY(...)` predicate is pushed into *both* legs by
the same code (retrieval/filter.py); the vector leg shares it.

    python scripts/build_governance_demo.py     # seed corpus + principals first
    python scripts/governance_eval.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sourcerer.config import get_settings
from sourcerer.governance import gate
from sourcerer.governance.principal import ANONYMOUS, get_principal
from sourcerer.retrieval import bm25


def _retrieved_sources(query: str, k: int, allowed: set[str] | None) -> set[str]:
    return {c.source for c in bm25.search(query, k, allowed)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the governance leakage eval.")
    parser.add_argument(
        "--set", default="eval/governance_eval_set.jsonl", help="Path to the eval set (jsonl)."
    )
    parser.add_argument("--k", type=int, default=5, help="top-k retrieved per query.")
    args = parser.parse_args()

    base = get_settings()
    gov = base.model_copy(update={"governance_enabled": True, "governance_pdp": "local"})

    rows = [json.loads(line) for line in Path(args.set).read_text().splitlines() if line.strip()]

    n_unauth = n_auth = 0
    leak_gated = leak_base = hit_gated = hit_base = 0
    per_principal: dict[str, dict] = {}

    print(f"{'principal':10} {'auth':5} {'src':12} {'baseline':9} {'gated':6}  query")
    print("-" * 78)
    for r in rows:
        principal = get_principal(r["principal"]) or ANONYMOUS
        allowed = gate.allowed_document_sources(principal, gov)  # gated visible set
        src = r["answer_source"]
        got_gated = src in _retrieved_sources(r["query"], args.k, allowed)
        got_base = src in _retrieved_sources(r["query"], args.k, None)

        stats = per_principal.setdefault(r["principal"], dict(leak=0, unauth=0, hit=0, auth=0))
        if r["authorized"]:
            n_auth += 1
            stats["auth"] += 1
            hit_gated += got_gated
            hit_base += got_base
            stats["hit"] += got_gated
            verdict = "HIT" if got_gated else "miss"
        else:
            n_unauth += 1
            stats["unauth"] += 1
            leak_gated += got_gated
            leak_base += got_base
            stats["leak"] += got_gated
            verdict = "LEAK!" if got_gated else "blocked"

        base_mark = "found" if got_base else "absent"
        print(
            f"{r['principal']:10} {str(r['authorized']):5} {src:12} "
            f"{base_mark:9} {verdict:6}  {r['query']}"
        )

    print("-" * 78)
    print("\n== Headline ==")
    print(f"Unauthorized cases:        {n_unauth}")
    print(f"  leakage (baseline/off):  {leak_base}/{n_unauth} = {_pct(leak_base, n_unauth)}")
    print(
        f"  leakage (gated):         {leak_gated}/{n_unauth} = {_pct(leak_gated, n_unauth)}"
        f"   {'✅ 0 leaks' if leak_gated == 0 else '❌ LEAK'}"
    )
    print(f"Authorized cases:          {n_auth}")
    print(f"  recall@{args.k} (baseline):    {hit_base}/{n_auth} = {_pct(hit_base, n_auth)}")
    print(
        f"  recall@{args.k} (gated):       {hit_gated}/{n_auth} = {_pct(hit_gated, n_auth)}"
        f"   {'✅ preserved' if hit_gated >= hit_base else '⚠ dropped'}"
    )

    print("\n== Per principal (gated) ==")
    for pid, s in per_principal.items():
        leak = f"{s['leak']}/{s['unauth']}" if s["unauth"] else "-"
        recall = f"{s['hit']}/{s['auth']}" if s["auth"] else "-"
        print(f"  {pid:10} leakage={leak:6} recall={recall}")


def _pct(n: int, d: int) -> str:
    return f"{(100 * n / d):.0f}%" if d else "n/a"


if __name__ == "__main__":
    main()
