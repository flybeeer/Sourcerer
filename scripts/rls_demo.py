"""Prove the Postgres RLS backstop (Phase 10d, defense-in-depth).

The point of RLS is independence from the app: it must block forbidden rows even
when the application adds *no* filter at all. This script runs `SELECT ... FROM
chunks` with no WHERE clause under the restricted reader role for each principal
and checks that only authorized sources come back — the data plane enforcing the
policy on its own.

Prereqs: `python scripts/build_governance_demo.py` then `python scripts/setup_rls.py`.

    python scripts/rls_demo.py
"""

from __future__ import annotations

from sourcerer.db.session import connect

# The tagged demo sources (build_governance_demo.py) and who should see each.
TAGGED = ["faq.md", "security.md", "handbook.md", "board.md"]
EXPECTED = {
    "anonymous": {"faq.md"},
    "alice": {"faq.md", "security.md"},
    "bob": {"faq.md", "security.md", "handbook.md"},
    "carol": {"faq.md", "security.md", "handbook.md", "board.md"},
}


def _reader_visible(principal: str) -> set[str]:
    """Sources visible to `principal` via RLS alone (no app-level WHERE)."""
    with connect(reader_principal=principal) as conn:
        seen = {r[0] for r in conn.execute("SELECT DISTINCT source FROM chunks").fetchall()}
    return {s for s in seen if s in TAGGED}


def main() -> None:
    # Owner bypasses RLS — confirms the rows are actually there to be filtered.
    with connect() as conn:
        owner = {r[0] for r in conn.execute("SELECT DISTINCT source FROM chunks").fetchall()}
    print("owner (RLS bypass) sees all tagged:", sorted(s for s in owner if s in TAGGED))
    print("\nReader role + RLS, NO app filter — the database enforces alone:")
    print(f"{'principal':10} {'visible':40} verdict")
    print("-" * 70)

    leaks = 0
    for principal, expected in EXPECTED.items():
        visible = _reader_visible(principal)
        forbidden = visible - expected
        leaks += len(forbidden)
        verdict = "✅ ok" if not forbidden else f"❌ LEAK {sorted(forbidden)}"
        print(f"{principal:10} {str(sorted(visible)):40} {verdict}")

    print("-" * 70)
    print(
        f"\nLeaked rows past RLS (app filter disabled): {leaks}  "
        f"{'✅ backstop holds' if leaks == 0 else '❌'}"
    )


if __name__ == "__main__":
    main()
