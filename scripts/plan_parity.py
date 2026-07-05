"""Three-way parity for the document gate: local vs CheckResources vs PlanResources.

The document policy is meant to give the *same* allow/deny no matter which PDP
path computes it:

  - local       — policy.can_read in-process (the written reference)
  - cerbos      — Cerbos CheckResources (enumerate assets → keep allowed ids)
  - cerbos-plan — Cerbos PlanResources → SQL predicate pushdown (governance/plan.py)

This script asserts all three return the identical visible-source set for every
seeded principal. It's the parity guard for the PlanResources work, mirroring the
"Cerbos parity 6/6" check from 10b — kept as a script (not pytest) because it
needs the live Cerbos sidecar + Postgres.

Run (with the demo seeded, sidecar up, GOVERNANCE_* exported):

    python scripts/build_governance_demo.py
    python scripts/plan_parity.py
"""

from __future__ import annotations

import sys

from sourcerer.config import get_settings
from sourcerer.governance import gate, plan
from sourcerer.governance.principal import resolve

PRINCIPAL_IDS = ["anonymous", "alice", "bob", "carol"]


def _local_set(principal, settings) -> set[str]:
    """Reference set: LocalPDP over the same document assets the gate enumerates."""
    from sourcerer.governance.gate import _document_assets
    from sourcerer.governance.policy import readable_assets

    return readable_assets(principal, _document_assets(settings))


def main() -> int:
    settings = get_settings()
    if not settings.governance_enabled:
        print("GOVERNANCE_ENABLED is off — export it (and GOVERNANCE_PDP=cerbos) first.")
        return 2

    check_settings = settings.model_copy(update={"governance_pdp": "cerbos"})
    ok = True
    print(f"{'principal':10} {'local':28} check  plan")
    for pid in PRINCIPAL_IDS:
        principal = resolve(pid)
        local = _local_set(principal, settings)
        check = gate.allowed_document_sources(principal, check_settings)
        plan_set = plan.allowed_document_sources_plan(principal, check_settings)

        agree = local == check == plan_set
        ok = ok and agree
        mark = "OK " if agree else "MISMATCH"
        print(f"{pid:10} {sorted(local)!s:28} {mark}")
        if not agree:
            print(f"           check={sorted(check)}  plan={sorted(plan_set)}")

    print("\nPARITY:", "PASS — all three PDP paths agree" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
