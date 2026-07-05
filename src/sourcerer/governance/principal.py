"""Principal identity — who is asking (Phase 10a).

A *principal* carries the RBAC role(s) and ABAC attributes (clearance, teams,
region) that the Cerbos gate will weigh against an asset's attributes from 10b
on. 10a only resolves and *records* the principal — no access is enforced yet.

Identity is a stub IdP: the request carries a principal id in a header
(`PRINCIPAL_HEADER`, default `X-Principal`); we look it up in a small
`principals` table. An unknown or absent id resolves to `ANONYMOUS` (public
clearance, no roles) — the least-privileged default, so a missing header can
never widen access.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sourcerer.db.session import connect
from sourcerer.governance.catalog import CLASSIFICATIONS, classification_rank


@dataclass(frozen=True)
class Principal:
    """An identity making a request, with its RBAC role(s) + ABAC attributes."""

    id: str
    roles: list[str] = field(default_factory=list)
    clearance: str = "public"
    teams: list[str] = field(default_factory=list)
    region: str | None = None

    @property
    def clearance_rank(self) -> int:
        """Rank of this principal's clearance on the classification lattice."""
        return classification_rank(self.clearance)

    @property
    def readable_classifications(self) -> list[str]:
        """The classification levels this clearance can read (lattice ≤ clearance).

        Precomputes the lattice comparison as an explicit set so the Cerbos policy
        can be `R.attr.classification in P.attr.readable_classifications` instead of
        rank arithmetic over a map — which keeps PlanResources' emitted plan a plain
        `IN`, translatable to a SQL `= ANY(...)` (see governance/plan.py). Equivalent
        to `clearance_rank >= classification_rank(c)` in policy.can_read, so local /
        Cerbos / plan PDPs stay in parity. Unknown classifications (rank > lattice)
        are excluded — fail-safe, matching can_read.
        """
        return [c for c in CLASSIFICATIONS if self.clearance_rank >= classification_rank(c)]

    def attr(self) -> dict:
        """Attribute bag shaped for a Cerbos `principal.attr` (used from 10b)."""
        return {
            "clearance": self.clearance,
            "teams": self.teams,
            "region": self.region,
            "readable_classifications": self.readable_classifications,
        }


# Least-privileged fallback for a missing/unknown principal: public clearance,
# no roles, no team. Fail-closed — an absent header must never grant more access.
ANONYMOUS = Principal(id="anonymous", roles=[], clearance="public", teams=[], region=None)


def init_principal_schema() -> None:
    """Create the `principals` table if absent (idempotent stub IdP)."""
    with connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS principals (
                id        TEXT NOT NULL PRIMARY KEY,
                roles     TEXT[] NOT NULL DEFAULT '{}',
                clearance TEXT   NOT NULL DEFAULT 'public',
                teams     TEXT[] NOT NULL DEFAULT '{}',
                region    TEXT
            )
            """)


def upsert_principal(principal: Principal) -> None:
    """Insert or update one principal (keyed on id)."""
    init_principal_schema()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO principals (id, roles, clearance, teams, region)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                roles     = EXCLUDED.roles,
                clearance = EXCLUDED.clearance,
                teams     = EXCLUDED.teams,
                region    = EXCLUDED.region
            """,
            (
                principal.id,
                principal.roles,
                principal.clearance,
                principal.teams,
                principal.region,
            ),
        )


def get_principal(principal_id: str) -> Principal | None:
    """Look up a principal by id, or None if unknown."""
    with connect() as conn:
        row = conn.execute(
            "SELECT id, roles, clearance, teams, region FROM principals WHERE id = %s",
            (principal_id,),
        ).fetchone()
    if not row:
        return None
    return Principal(
        id=row[0],
        roles=list(row[1]) if row[1] else [],
        clearance=row[2],
        teams=list(row[3]) if row[3] else [],
        region=row[4],
    )


def list_principals() -> list[Principal]:
    """List all known principals (the stub IdP's directory)."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, roles, clearance, teams, region FROM principals ORDER BY id"
        ).fetchall()
    return [
        Principal(
            id=r[0],
            roles=list(r[1]) if r[1] else [],
            clearance=r[2],
            teams=list(r[3]) if r[3] else [],
            region=r[4],
        )
        for r in rows
    ]


def resolve(principal_id: str | None) -> Principal:
    """Resolve a header value to a Principal, falling back to ANONYMOUS.

    Fail-closed: a missing header, an unknown id, or a catalog that hasn't been
    set up all resolve to the least-privileged `ANONYMOUS` principal.
    """
    if not principal_id:
        return ANONYMOUS
    try:
        found = get_principal(principal_id.strip())
    except Exception:  # principals table not provisioned yet → least privilege
        return ANONYMOUS
    return found or ANONYMOUS
