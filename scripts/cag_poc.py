"""PoC — Cache-Augmented Generation (CAG) governed by Cerbos.

Demonstrates the integration point: instead of retrieving per query (RAG), the
authorized corpus is preloaded into a per-access-profile context cache, and the
**Cerbos gate decides cache membership at build time**. Forbidden documents never
enter a principal's cache, so the Phase 10 rule holds without any per-query gate.

Shows:
  1. Cerbos partitions caches — each principal's cache holds only readable sources.
  2. Same question, different principals → different answers (governance preserved,
     no retrieval step).
  3. Cache reuse — a principal's cache is built once and reused across questions
     (the CAG win: no re-embed / re-search, and a stable context prefix that a
     KV-caching backend reuses).

Prereqs: the tagged demo corpus + principals (see docs/governance-phase-10-demo.md)
and a running Cerbos sidecar. Run:  python scripts/cag_poc.py
"""

from __future__ import annotations

import time

from sourcerer.cag import cache as cag_cache
from sourcerer.cag import generate as cag_gen
from sourcerer.config import get_settings
from sourcerer.governance.principal import get_principal
from sourcerer.llm.client import get_llm_client

# Keep the PoC corpus small enough to fit one context (CAG's premise).
DEMO = {"faq.md", "security.md", "handbook.md", "board.md"}
WHO = ["anonymous", "alice", "bob", "carol"]


def main() -> None:
    settings = get_settings().model_copy(
        update={"governance_enabled": True, "governance_pdp": "cerbos"}
    )
    client = get_llm_client(settings)
    cag_cache.clear()

    print("=" * 74)
    print("1) Cerbos partitions the cache at BUILD time (forbidden docs never load)")
    print("=" * 74)
    caches = {}
    for who in WHO:
        p = get_principal(who)
        c = cag_cache.build_cache(p, settings, limit_sources=DEMO)
        caches[who] = c
        print(f"  {who:10} cache sources = {c.sources}")

    print("\n" + "=" * 74)
    print("2) Same question, no retrieval — answered from each cache")
    print("   Q: 'What are the employee salary bands?'  (handbook.md = confidential/hr)")
    print("=" * 74)
    for who in WHO:
        ans = cag_gen.answer("What are the employee salary bands?", caches[who], client)
        has = "handbook.md" in ans.sources_in_context
        print(f"  {who:10} [handbook in cache: {str(has):5}] → {ans.text[:90].strip()}")

    print("\n" + "=" * 74)
    print("3) Cache reuse — bob asks 3 questions; cache built once, reused")
    print("=" * 74)
    bob = get_principal("bob")
    questions = [
        "What is the office opening hours?",
        "What is the password rotation policy?",
        "What are the employee salary bands?",
    ]
    for q in questions:
        t0 = time.perf_counter()
        c = cag_cache.build_cache(bob, settings, limit_sources=DEMO)  # reuse, not rebuild
        ans = cag_gen.answer(q, c, client)
        dt = (time.perf_counter() - t0) * 1000
        print(f"  [cache hits={c.hits}] {dt:6.0f}ms  Q: {q[:38]:38} → {ans.text[:60].strip()}")

    print(
        f"\n  bob's cache was built once (build_ms={caches['bob'].build_ms}) and reused "
        f"{caches['bob'].hits + len(questions)}×."
    )
    print("  No per-query retrieval/embedding; governance enforced once, at build time.")


if __name__ == "__main__":
    main()
